"""Conversion utilities between Flax Linen and NNX modules."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
import flax
from plstm.nnx_dummy import nnx

from ..linen.util import module_named_params as linen_module_named_params
from ..linen.util import count_parameters as count_parameters_linen
from ..nnx.util import module_named_params as nnx_module_named_params
from ..nnx.util import count_parameters as count_parameters_nnx
from .base import ConversionRegistry, _get_param_names_and_shape_linen, _get_param_names_and_shape_nnx


def module_paths(
    module: nn.Module,
    example_inputs: Any,
    rng_key: jax.random.PRNGKey = None,
) -> dict[str, nn.Module]:
    """Traverse `module` (a Flax nn.Module) with `example_inputs` and return a
    dict mapping each submodule's string path to the Module instance."""
    try:
        return module.module_paths(rng_key, example_inputs)
    except (AttributeError, flax.errors.InvalidRngError):
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # 1. Initialize parameters so that all submodules get constructed
        variables = module.init(rng_key, example_inputs)

        # 2. Set up capture hook
        seen = set()
        modules: list[tuple[nn.Module, tuple[str, ...]]] = []

        def _capture_fn(mod: nn.Module, method_name: str) -> bool:
            # record each module exactly once
            if id(mod) not in seen:
                modules.append((mod, mod.path))
                seen.add(id(mod))
            # return False so that Flax does _not_ actually store intermediate outputs
            return False

        # 3. Run a no‑op shape inference pass to fire the hook
        jax.eval_shape(lambda: module.apply(variables, example_inputs, capture_intermediates=_capture_fn))

        # print(modules)
        # 4. Build a dict from joined‐path → module
        return {"/".join(path): mod for (mod, path) in modules}


def _default_conversion_linen_to_nnx(
    linen_module: nn.Module,
    nnx_module: nnx.Module,
    variables: dict,
    linen_children: dict[str, nn.Module] = {},
    strict: bool = True,
) -> None:
    """Default parameter conversion from Linen to NNX. Only converts top level
    parameters.

    Args:
        linen_module: Source Linen module
        nnx_module: Target NNX module to update
        variables: Linen variables dict
        strict: Whether to raise an error if parameters don't match
    """
    # Get parameters
    linen_params = dict(linen_module_named_params(linen_module, variables, recursive=False))
    nnx_params = dict(nnx_module_named_params(nnx_module, recursive=False))

    # Copy matching parameters
    for name, linen_param in linen_params.items():
        if name in nnx_params:
            setattr(nnx_module, name, nnx.Param(jnp.array(linen_param)))
        elif strict:
            raise ValueError(f"Parameter {name} not found in nnx module")


def _default_conversion_nnx_to_linen(
    nnx_module: nnx.Module,
    linen_module: nn.Module,
    variables: dict,
    linen_children: dict[str, nn.Module] = {},
    strict: bool = True,
) -> dict:
    """Default parameter conversion from NNX to Linen. Only converts top level
    paramters.

    Args:
        nnx_module: Source NNX module
        linen_module: Target Linen module
        variables: Linen variables dict
        strict: Whether to raise an error if parameters don't match

    Returns:
        Updated variables dict
    """
    # Get parameters
    nnx_params = dict(nnx_module_named_params(nnx_module, recursive=False))
    linen_params = dict(linen_module_named_params(linen_module, variables, recursive=False))

    # Create updated variables
    updated_variables = variables.copy()

    for name, nnx_param in nnx_params.items():
        if name in linen_params:
            # Convert parameter path to tuple format for flax
            param = jnp.array(nnx_param)
            updated_variables["params"][name] = param
        elif strict:
            raise ValueError(f"Parameter {name} not found in linen module")

    return updated_variables


def convert_parameters_linen_to_nnx(
    linen_module: nn.Module,
    nnx_module: nnx.Module,
    variables: dict,
    exmp_input: Any = None,
    linen_children: dict | None = None,
    strict: bool = True,
) -> None:
    """Convert parameters from a Linen module to an NNX module.

    Args:
        linen_module: Source Linen module
        nnx_module: Target NNX module
        variables: Linen variables dict
        strict: Whether to raise an error if parameters don't match
    """
    assert count_parameters_linen(linen_module, variables) == count_parameters_nnx(nnx_module), (
        f"Different number of parameters: {_get_param_names_and_shape_linen(linen_module, variables)} != "
        f"{_get_param_names_and_shape_nnx(nnx_module, variables)}"
    )

    # Get conversion function for this module type if registered
    conversion_fn = ConversionRegistry.get_conversion(type(linen_module), type(nnx_module))

    if exmp_input is not None:
        linen_children = module_paths(linen_module, exmp_input, rng_key=jax.random.PRNGKey(0))
    else:
        assert (
            linen_children is not None
        ), "For linen conversions either all modules as linen_children or an exemplary input have to be given."

    # Convert parameters
    if conversion_fn is not None:
        # Use registered conversion
        conversion_fn(linen_module, nnx_module, variables=variables, linen_children=linen_children)
    else:
        # Use default conversion for this module
        _default_conversion_linen_to_nnx(
            linen_module, nnx_module, variables, linen_children=linen_children, strict=strict
        )

        # Handle child modules recursively
        for name, linen_child in linen_children.items():
            if (
                name
                and "/" not in name
                and hasattr(nnx_module, name)
                and isinstance(getattr(nnx_module, name), nnx.Module)
            ):
                # Get the corresponding child variables
                child_variables = {"params": variables["params"].get(name, {})}
                for collection in variables:
                    if collection != "params" and name in variables[collection]:
                        child_variables[collection] = variables[collection][name]
                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(name + "/")
                }
                # Recursively convert child modules
                convert_parameters_linen_to_nnx(
                    linen_child,
                    getattr(nnx_module, name),
                    child_variables,
                    linen_children=linen_sub_children,
                    strict=strict,
                )


def convert_parameters_nnx_to_linen(
    nnx_module: nnx.Module,
    linen_module: nn.Module,
    variables: dict,
    exmp_input: Any = None,
    linen_children: dict | None = None,
    strict: bool = True,
) -> dict:
    """Convert parameters from an NNX module to a Linen module.

    Args:
        nnx_module: Source NNX module
        linen_module: Target Linen module
        variables: Linen variables dict
        strict: Whether to raise an error if parameters don't match

    Returns:
        Updated variables dict
    """
    assert count_parameters_linen(linen_module, variables) == count_parameters_nnx(nnx_module), (
        f"Different number of parameters: {_get_param_names_and_shape_linen(linen_module, variables)} != "
        f"{_get_param_names_and_shape_nnx(nnx_module, variables)}"
    )

    # Get conversion function for this module type if registered
    conversion_fn = ConversionRegistry.get_conversion(type(nnx_module), type(linen_module))

    if exmp_input is not None:
        linen_children = module_paths(linen_module, exmp_input, rng_key=jax.random.PRNGKey(0))
    else:
        assert (
            linen_children is not None
        ), "For linen conversions either all modules as linen_children or an exemplary input have to be given."

    # Convert parameters
    updated_variables = variables.copy()
    if conversion_fn is not None:
        # Use registered conversion
        updated_variables = conversion_fn(nnx_module, linen_module, variables=variables, linen_children=linen_children)
    else:
        # Use default conversion for this module
        updated_variables = _default_conversion_nnx_to_linen(
            nnx_module, linen_module, variables, linen_children=linen_children, strict=strict
        )

        for name, linen_child in linen_children.items():
            if (
                name
                and "/" not in name
                and hasattr(nnx_module, name)
                and isinstance(getattr(nnx_module, name), nnx.Module)
            ):
                # Get the corresponding child variables
                child_variables = {"params": updated_variables["params"].get(name, {})}
                for collection in updated_variables:
                    if collection != "params" and name in updated_variables[collection]:
                        child_variables[collection] = updated_variables[collection][name]

                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(name + "/")
                }

                # Recursively convert child modules
                updated_child_variables = convert_parameters_nnx_to_linen(
                    getattr(nnx_module, name),
                    linen_child,
                    child_variables,
                    linen_children=linen_sub_children,
                    strict=strict,
                )

                # Update the parent variables with the child variables
                updated_variables["params"][name] = updated_child_variables["params"]
                for collection in updated_child_variables:
                    if collection != "params":
                        if collection not in updated_variables:
                            updated_variables[collection] = {}
                        updated_variables[collection][name] = updated_child_variables[collection]

    return updated_variables


# Register common module conversions between Linen and NNX


@ConversionRegistry.register(nn.Dense, nnx.Linear)
def convert_dense_linear(linen_module: nn.Dense, nnx_module: nnx.Linear, *, reverse: bool = False, **kwargs):
    """Convert parameters between Linen Dense and NNX Linear modules."""
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # NNX to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Update kernel parameter
        params["kernel"] = jnp.array(nnx_module.kernel)

        # Update bias if present
        if hasattr(nnx_module, "bias") and nnx_module.bias is not None and nnx_module.bias.value is not None:
            params["bias"] = jnp.array(nnx_module.bias)

        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to NNX
        kernel = variables["params"]["kernel"]
        nnx_module.kernel = nnx.Param(jnp.array(kernel))

        if "bias" in variables["params"]:
            bias = variables["params"]["bias"]
            nnx_module.bias = nnx.Param(jnp.array(bias))


@ConversionRegistry.register(nn.LayerNorm, nnx.LayerNorm)
def convert_layernorm(linen_module: nn.LayerNorm, nnx_module: nnx.LayerNorm, *, reverse: bool = False, **kwargs):
    """Convert parameters between Linen LayerNorm and NNX LayerNorm modules."""
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # NNX to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Update scale parameter
        if hasattr(nnx_module, "scale") and nnx_module.scale is not None and nnx_module.scale.value is not None:
            params["scale"] = jnp.array(nnx_module.scale)

        # Update bias if present
        if hasattr(nnx_module, "bias") and nnx_module.bias is not None and nnx_module.bias.value is not None:
            params["bias"] = jnp.array(nnx_module.bias)

        if params:
            updated_variables["params"] = params
        return updated_variables
    else:
        if variables:
            # Linen to NNX
            if "scale" in variables["params"]:
                scale = variables["params"]["scale"]
                nnx_module.scale = nnx.Param(jnp.array(scale))

            if "bias" in variables["params"]:
                bias = variables["params"]["bias"]
                nnx_module.bias = nnx.Param(jnp.array(bias))


@ConversionRegistry.register(nn.Embed, nnx.Embed)
def convert_embedding(linen_module: nn.Embed, nnx_module: nnx.Embed, *, reverse: bool = False, **kwargs):
    """Convert parameters between Linen Embed and NNX Embed modules."""
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # NNX to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Update embedding parameter
        if (
            hasattr(nnx_module, "embedding")
            and nnx_module.embedding is not None
            and nnx_module.embedding.value is not None
        ):
            params["embedding"] = jnp.array(nnx_module.embedding)

        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to NNX
        if "embedding" in variables["params"]:
            embedding = variables["params"]["embedding"]
            nnx_module.embedding = nnx.Param(jnp.array(embedding))


@ConversionRegistry.register(nn.Conv, nnx.Conv)
def convert_conv(linen_module: nn.Conv, nnx_module: nnx.Conv, *, reverse: bool = False, **kwargs):
    """Convert parameters between Linen Conv and NNX Conv modules."""
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # NNX to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Update kernel parameter
        if hasattr(nnx_module, "kernel") and nnx_module.kernel is not None and nnx_module.kernel.value is not None:
            params["kernel"] = jnp.array(nnx_module.kernel)

        # Update bias if present
        if hasattr(nnx_module, "bias") and nnx_module.bias is not None and nnx_module.bias.value is not None:
            params["bias"] = jnp.array(nnx_module.bias)

        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to NNX
        if "kernel" in variables["params"]:
            kernel = variables["params"]["kernel"]
            nnx_module.kernel = nnx.Param(jnp.array(kernel))

        if "bias" in variables["params"]:
            bias = variables["params"]["bias"]
            nnx_module.bias = nnx.Param(jnp.array(bias))
