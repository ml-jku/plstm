"""Conversion utilities between Flax Linen and PyTorch modules."""

from typing import Any

import torch
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from ..linen.util import module_named_params as linen_module_named_params
from ..linen.util import count_parameters as count_parameters_linen
from ..torch.util import count_parameters as count_parameters_torch
from .base import ConversionRegistry, _get_param_names_and_shape_linen, _get_param_names_and_shape_torch
from .linen_nnx import module_paths


def _convert_to_torch(ar: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a torch tensor."""
    if ar.dtype == np.dtype("bfloat16"):
        return torch.from_numpy(ar.astype(np.float32)).to(torch.bfloat16)
    else:
        return torch.from_numpy(ar)


def _convert_to_jax(tensor: torch.Tensor) -> jnp.ndarray:
    """Convert a torch tensor to a jax array."""
    return jnp.array(tensor.detach().cpu().numpy())


def _default_conversion_linen_to_torch(
    linen_module: nn.Module,
    torch_module: torch.nn.Module,
    variables: dict,
    linen_children: dict[str, nn.Module] = {},
    strict: bool = True,
) -> None:
    """Default parameter conversion from Linen to PyTorch. Only converts top
    level parameters.

    Args:
        linen_module: Source Linen module
        torch_module: Target PyTorch module to update
        variables: Linen variables dict
        strict: Whether to raise an error if parameters don't match
    """
    # Get parameters for this module level
    linen_params = dict(linen_module_named_params(linen_module, variables, recursive=False))
    torch_params = dict(torch_module.named_parameters(recurse=False))

    # Copy matching parameters
    for name, linen_param in linen_params.items():
        if name in torch_params:
            with torch.no_grad():
                torch_params[name].copy_(_convert_to_torch(np.array(linen_param)))
        elif strict:
            raise ValueError(f"Parameter {name} not found in torch module")


def _default_conversion_torch_to_linen(
    torch_module: torch.nn.Module,
    linen_module: nn.Module,
    variables: dict,
    linen_children: dict[str, nn.Module] = {},
    strict: bool = True,
) -> dict:
    """Default parameter conversion from PyTorch to Linen. Only converts top
    level parameters.

    Args:
        torch_module: Source PyTorch module
        linen_module: Target Linen module
        variables: Linen variables dict
        strict: Whether to raise an error if parameters don't match

    Returns:
        Updated variables dict
    """
    # Get parameters
    torch_params = dict(torch_module.named_parameters(recurse=False))
    linen_params = dict(linen_module_named_params(linen_module, variables, recursive=False))

    # Create updated variables
    updated_variables = variables.copy()

    for name, torch_param in torch_params.items():
        if name in linen_params:
            # Convert parameter
            param = _convert_to_jax(torch_param)
            updated_variables["params"][name] = param
        elif strict:
            raise ValueError(f"Parameter {name} not found in linen module")

    return updated_variables


def convert_parameters_linen_to_torch(
    linen_module: nn.Module,
    torch_module: torch.nn.Module,
    variables: dict,
    exmp_input: Any = None,
    linen_children: dict | None = None,
    strict: bool = True,
) -> None:
    """Convert parameters from a Linen module to a PyTorch module.

    Args:
        linen_module: Source Linen module
        torch_module: Target PyTorch module to update
        variables: Linen variables dict
        strict: Whether to raise an error if parameters don't match
    """
    assert count_parameters_linen(linen_module, variables) == count_parameters_torch(torch_module), (
        f"Different number of parameters: {_get_param_names_and_shape_torch(torch_module)} != "
        f"{_get_param_names_and_shape_linen(linen_module, variables)}"
    )

    # Get conversion function for this module type if registered
    conversion_fn = ConversionRegistry.get_conversion(type(linen_module), type(torch_module))

    if exmp_input is not None:
        linen_children = module_paths(linen_module, exmp_input)
    else:
        assert (
            linen_children is not None
        ), "For linen conversions either all modules as linen_children or an exemplary input have to be given."

    # Convert parameters
    if conversion_fn is not None:
        # Use registered conversion
        conversion_fn(linen_module, torch_module, variables=variables, linen_children=linen_children)
    else:
        # Use default conversion for this module
        _default_conversion_linen_to_torch(
            linen_module, torch_module, variables, linen_children=linen_children, strict=strict
        )

        # Handle child modules recursively
        for name, linen_child in linen_children.items():
            if (
                name
                and "/" not in name
                and hasattr(torch_module, name)
                and isinstance(getattr(torch_module, name), torch.nn.Module)
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
                convert_parameters_linen_to_torch(
                    linen_child,
                    getattr(torch_module, name),
                    child_variables,
                    linen_children=linen_sub_children,
                    strict=strict,
                )


def convert_parameters_torch_to_linen(
    torch_module: torch.nn.Module,
    linen_module: nn.Module,
    variables: dict,
    exmp_input: Any = None,
    linen_children: dict | None = None,
    strict: bool = True,
) -> dict:
    """Convert parameters from a PyTorch module to a Linen module.

    Args:
        torch_module: Source PyTorch module
        linen_module: Target Linen module
        variables: Linen variables dict
        strict: Whether to raise an error if parameters don't match

    Returns:
        Updated variables dict
    """
    assert count_parameters_torch(torch_module) == count_parameters_linen(linen_module, variables), (
        f"Different number of parameters: {_get_param_names_and_shape_torch(torch_module)} != "
        f"{_get_param_names_and_shape_linen(linen_module, variables)}"
    )

    # Get conversion function for this module type if registered
    conversion_fn = ConversionRegistry.get_conversion(type(torch_module), type(linen_module))

    if exmp_input is not None:
        linen_children = module_paths(linen_module, exmp_input)
    else:
        assert (
            linen_children is not None
        ), "For linen conversions either all modules as linen_children or an exemplary input have to be given."

    # Convert parameters
    updated_variables = variables.copy()
    if conversion_fn is not None:
        # Use registered conversion
        updated_variables = conversion_fn(
            torch_module, linen_module, variables=variables, linen_children=linen_children
        )
    else:
        # Use default conversion for this module
        updated_variables = _default_conversion_torch_to_linen(
            torch_module, linen_module, variables, linen_children=linen_children, strict=strict
        )

        # Handle child modules recursively
        for name, linen_child in linen_children.items():
            if (
                name
                and "/" not in name
                and hasattr(torch_module, name)
                and isinstance(getattr(torch_module, name), torch.nn.Module)
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
                updated_child_variables = convert_parameters_torch_to_linen(
                    getattr(torch_module, name),
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


# Register common module conversions between Linen and PyTorch


@ConversionRegistry.register(nn.Dense, torch.nn.Linear)
def convert_dense_linear(linen_module: nn.Dense, torch_module: torch.nn.Linear, *, reverse: bool = False, **kwargs):
    """Convert parameters between Linen Dense and PyTorch Linear modules."""
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # PyTorch to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Update kernel parameter (transpose to match Linen format)
        kernel = torch_module.weight.cpu().detach().numpy().transpose(1, 0)
        params["kernel"] = jnp.array(kernel)

        # Update bias if present
        if hasattr(torch_module, "bias") and torch_module.bias is not None:
            bias = torch_module.bias.detach().cpu().numpy()
            params["bias"] = jnp.array(bias)

        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to PyTorch
        # Get the kernel parameter from variables
        kernel = variables["params"]["kernel"]
        with torch.no_grad():
            torch_module.weight.copy_(_convert_to_torch(np.array(kernel)).transpose(1, 0))

            # Handle bias if present
            if "bias" in variables["params"]:
                bias = variables["params"]["bias"]
                torch_module.bias.copy_(_convert_to_torch(np.array(bias)))


@ConversionRegistry.register(nn.LayerNorm, torch.nn.LayerNorm)
def convert_layernorm(linen_module: nn.LayerNorm, torch_module: torch.nn.LayerNorm, *, reverse: bool = False, **kwargs):
    """Convert parameters between Linen LayerNorm and PyTorch LayerNorm
    modules."""
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # PyTorch to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Update scale parameter
        if hasattr(torch_module, "weight") and torch_module.weight is not None:
            params["scale"] = jnp.array(torch_module.weight.cpu().detach().numpy())

        # Update bias if present
        if hasattr(torch_module, "bias") and torch_module.bias is not None:
            params["bias"] = jnp.array(torch_module.bias.cpu().detach().numpy())

        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to PyTorch
        with torch.no_grad():
            if variables:
                # Get the scale parameter from variables
                if "scale" in variables["params"]:
                    scale = variables["params"]["scale"]
                    torch_module.weight.copy_(_convert_to_torch(np.array(scale)))

                # Handle bias if present
                if "bias" in variables["params"]:
                    bias = variables["params"]["bias"]
                    torch_module.bias.copy_(_convert_to_torch(np.array(bias)))


@ConversionRegistry.register(nn.Embed, torch.nn.Embedding)
def convert_embedding(linen_module: nn.Embed, torch_module: torch.nn.Embedding, *, reverse: bool = False, **kwargs):
    """Convert parameters between Linen Embed and PyTorch Embedding modules."""
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # PyTorch to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Update embedding parameter
        if hasattr(torch_module, "weight") and torch_module.weight is not None:
            params["embedding"] = jnp.array(torch_module.weight.cpu().detach().numpy())

        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to PyTorch
        with torch.no_grad():
            # Get the embedding parameter from variables
            if "embedding" in variables["params"]:
                embedding = variables["params"]["embedding"]
                torch_module.weight.copy_(_convert_to_torch(np.array(embedding)))


@ConversionRegistry.register(nn.Conv, torch.nn.Conv1d)
def convert_conv1d(linen_module: nn.Conv, torch_module: torch.nn.Conv1d, *, reverse: bool = False, **kwargs):
    """Convert parameters between Linen Conv and PyTorch Conv1d modules."""
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # PyTorch to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Update kernel parameter (transpose to match Linen format)
        kernel = torch_module.weight.cpu().detach().numpy()
        # Transpose from [out_channels, in_channels, kernel_size] to [kernel_size, in_channels, out_channels]
        kernel = np.transpose(kernel, (2, 1, 0))
        params["kernel"] = jnp.array(kernel)

        # Update bias if present
        if hasattr(torch_module, "bias") and torch_module.bias is not None:
            bias = torch_module.bias.detach().cpu().numpy()
            params["bias"] = jnp.array(bias)

        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to PyTorch
        with torch.no_grad():
            # Get the kernel parameter from variables
            if "kernel" in variables["params"]:
                kernel = variables["params"]["kernel"]
                # Transpose from [kernel_size, in_channels, out_channels] to [out_channels, in_channels, kernel_size]
                kernel_np = np.array(kernel)
                kernel_np = np.transpose(kernel_np, (2, 1, 0))
                torch_module.weight.copy_(_convert_to_torch(kernel_np))

            # Handle bias if present
            if "bias" in variables["params"]:
                bias = variables["params"]["bias"]
                torch_module.bias.copy_(_convert_to_torch(np.array(bias)))


@ConversionRegistry.register(nn.Conv, torch.nn.Conv2d)
def convert_conv2d(linen_module: nn.Conv, torch_module: torch.nn.Conv2d, *, reverse: bool = False, **kwargs):
    """Convert parameters between Linen Conv and PyTorch Conv2d modules."""
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # PyTorch to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Update kernel parameter (transpose to match Linen format)
        kernel = torch_module.weight.cpu().detach().numpy()
        # Transpose from [out_channels, in_channels, kernel_height, kernel_width] to
        # [kernel_height, kernel_width, in_channels, out_channels]
        kernel = np.transpose(kernel, (2, 3, 1, 0))
        params["kernel"] = jnp.array(kernel)

        # Update bias if present
        if hasattr(torch_module, "bias") and torch_module.bias is not None:
            bias = torch_module.bias.detach().cpu().numpy()
            params["bias"] = jnp.array(bias)

        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to PyTorch
        with torch.no_grad():
            # Get the kernel parameter from variables
            if "kernel" in variables["params"]:
                kernel = variables["params"]["kernel"]
                # Transpose from [kernel_height, kernel_width, in_channels, out_channels] to
                # [out_channels, in_channels, kernel_height, kernel_width]
                kernel_np = np.array(kernel)
                kernel_np = np.transpose(kernel_np, (3, 2, 0, 1))
                torch_module.weight.copy_(_convert_to_torch(kernel_np))

            # Handle bias if present
            if "bias" in variables["params"]:
                bias = variables["params"]["bias"]
                torch_module.bias.copy_(_convert_to_torch(np.array(bias)))
