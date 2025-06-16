"""Parameter conversion utilities for normalization layers between JAX
(nnx/linen) and PyTorch."""

import torch
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx

from .base import ConversionRegistry, _convert_to_torch

# Import the implementation classes
from ..torch.norm import (
    MultiHeadLayerNorm as TorchMultiHeadLayerNorm,
    MultiHeadRMSNorm as TorchMultiHeadRMSNorm,
    LayerNorm as TorchLayerNorm,
    RMSNorm as TorchRMSNorm,
    Identity as TorchIdentity,
)

from ..nnx.norm import (
    MultiHeadLayerNorm as NNXMultiHeadLayerNorm,
    MultiHeadRMSNorm as NNXMultiHeadRMSNorm,
    LayerNorm as NNXLayerNorm,
    RMSNorm as NNXRMSNorm,
    Identity as NNXIdentity,
)

from ..linen.norm import (
    MultiHeadLayerNorm as LinenMultiHeadLayerNorm,
    MultiHeadRMSNorm as LinenMultiHeadRMSNorm,
    LayerNorm as LinenLayerNorm,
    RMSNorm as LinenRMSNorm,
    Identity as LinenIdentity,
)


# NNX <-> PyTorch conversions


@ConversionRegistry.register(NNXMultiHeadLayerNorm, TorchMultiHeadLayerNorm)
def convert_multihead_layernorm_nnx_torch(
    nnx_module: NNXMultiHeadLayerNorm, torch_module: TorchMultiHeadLayerNorm, *, reverse: bool = False, **kwargs
):
    """Convert parameters between NNX and PyTorch MultiHeadLayerNorm modules.

    Args:
        nnx_module: NNX MultiHeadLayerNorm module
        torch_module: PyTorch MultiHeadLayerNorm module
        reverse: If True, convert from PyTorch to NNX. If False, convert from NNX to PyTorch.
        **kwargs: Additional arguments (not used)
    """
    if reverse:
        # PyTorch to NNX
        if torch_module.scale is not None:
            nnx_module.norm.scale = nnx.Param(jnp.array(torch_module.scale.detach().cpu().numpy()))
        if torch_module.bias is not None:
            nnx_module.norm.bias = nnx.Param(jnp.array(torch_module.bias.detach().cpu().numpy()))
    else:
        # NNX to PyTorch
        if hasattr(nnx_module.norm, "scale") and torch_module.scale is not None:
            with torch.no_grad():
                torch_module.scale.copy_(_convert_to_torch(np.array(nnx_module.norm.scale)))
        if hasattr(nnx_module.norm, "bias") and torch_module.bias is not None:
            with torch.no_grad():
                torch_module.bias.copy_(_convert_to_torch(np.array(nnx_module.norm.bias)))


@ConversionRegistry.register(NNXMultiHeadRMSNorm, TorchMultiHeadRMSNorm)
def convert_multihead_rmsnorm_nnx_torch(
    nnx_module: NNXMultiHeadRMSNorm, torch_module: TorchMultiHeadRMSNorm, *, reverse: bool = False, **kwargs
):
    """Convert parameters between NNX and PyTorch MultiHeadRMSNorm modules.

    Args:
        nnx_module: NNX MultiHeadRMSNorm module
        torch_module: PyTorch MultiHeadRMSNorm module
        reverse: If True, convert from PyTorch to NNX. If False, convert from NNX to PyTorch.
        **kwargs: Additional arguments (not used)
    """
    if reverse:
        # PyTorch to NNX
        if torch_module.scale is not None:
            nnx_module.norm.scale = nnx.Param(jnp.array(torch_module.scale.detach().cpu().numpy()))
    else:
        # NNX to PyTorch
        if hasattr(nnx_module.norm, "scale") and torch_module.scale is not None:
            with torch.no_grad():
                torch_module.scale.copy_(_convert_to_torch(np.array(nnx_module.norm.scale)))


@ConversionRegistry.register(NNXLayerNorm, TorchLayerNorm)
def convert_layernorm_nnx_torch(
    nnx_module: NNXLayerNorm, torch_module: TorchLayerNorm, *, reverse: bool = False, **kwargs
):
    """Convert parameters between NNX and PyTorch LayerNorm modules.

    Args:
        nnx_module: NNX LayerNorm module
        torch_module: PyTorch LayerNorm module
        reverse: If True, convert from PyTorch to NNX. If False, convert from NNX to PyTorch.
        **kwargs: Additional arguments (not used)
    """
    if reverse:
        # PyTorch to NNX
        if torch_module.scale is not None:
            nnx_module.norm.scale = nnx.Param(jnp.array(torch_module.scale.detach().cpu().numpy()))
        if torch_module.bias is not None:
            nnx_module.norm.bias = nnx.Param(jnp.array(torch_module.bias.detach().cpu().numpy()))
    else:
        # NNX to PyTorch
        if hasattr(nnx_module.norm, "scale") and torch_module.scale is not None:
            with torch.no_grad():
                torch_module.scale.copy_(_convert_to_torch(np.array(nnx_module.norm.scale)))
        if hasattr(nnx_module.norm, "bias") and torch_module.bias is not None:
            with torch.no_grad():
                torch_module.bias.copy_(_convert_to_torch(np.array(nnx_module.norm.bias)))


@ConversionRegistry.register(NNXRMSNorm, TorchRMSNorm)
def convert_rmsnorm_nnx_torch(nnx_module: NNXRMSNorm, torch_module: TorchRMSNorm, *, reverse: bool = False, **kwargs):
    """Convert parameters between NNX and PyTorch RMSNorm modules.

    Args:
        nnx_module: NNX RMSNorm module
        torch_module: PyTorch RMSNorm module
        reverse: If True, convert from PyTorch to NNX. If False, convert from NNX to PyTorch.
        **kwargs: Additional arguments (not used)
    """
    if reverse:
        # PyTorch to NNX
        if torch_module.scale is not None:
            nnx_module.norm.scale = nnx.Param(jnp.array(torch_module.scale.detach().cpu().numpy()))
    else:
        # NNX to PyTorch
        if hasattr(nnx_module.norm, "scale") and torch_module.scale is not None:
            with torch.no_grad():
                torch_module.scale.copy_(_convert_to_torch(np.array(nnx_module.norm.scale)))


@ConversionRegistry.register(NNXIdentity, TorchIdentity)
def convert_identity_nnx_torch(
    nnx_module: NNXIdentity, torch_module: TorchIdentity, *, reverse: bool = False, **kwargs
):
    """Convert parameters between NNX and PyTorch Identity modules.

    This is a no-op since Identity has no parameters, but included for completeness.

    Args:
        nnx_module: NNX Identity module
        torch_module: PyTorch Identity module
        reverse: If True, convert from PyTorch to NNX. If False, convert from NNX to PyTorch.
        **kwargs: Additional arguments (not used)
    """
    # Identity has no parameters, so this is a no-op
    pass


# Linen <-> PyTorch conversions


@ConversionRegistry.register(LinenMultiHeadLayerNorm, TorchMultiHeadLayerNorm)
def convert_multihead_layernorm_linen_torch(
    linen_module: LinenMultiHeadLayerNorm, torch_module: TorchMultiHeadLayerNorm, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and PyTorch MultiHeadLayerNorm modules.

    Args:
        linen_module: Linen MultiHeadLayerNorm module
        torch_module: PyTorch MultiHeadLayerNorm module
        reverse: If True, convert from PyTorch to Linen. If False, convert from Linen to PyTorch.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # PyTorch to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Create norm parameters
        norm_params = {}
        if torch_module.scale is not None:
            norm_params["scale"] = jnp.array(torch_module.scale.detach().cpu().numpy())
        if torch_module.bias is not None:
            norm_params["bias"] = jnp.array(torch_module.bias.detach().cpu().numpy())

        params["norm"] = norm_params
        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to PyTorch
        with torch.no_grad():
            if variables:
                if "norm" in variables["params"]:
                    norm_params = variables["params"]["norm"]
                    if "scale" in norm_params and torch_module.scale is not None:
                        torch_module.scale.copy_(_convert_to_torch(np.array(norm_params["scale"])))
                    if "bias" in norm_params and torch_module.bias is not None:
                        torch_module.bias.copy_(_convert_to_torch(np.array(norm_params["bias"])))


@ConversionRegistry.register(LinenMultiHeadRMSNorm, TorchMultiHeadRMSNorm)
def convert_multihead_rmsnorm_linen_torch(
    linen_module: LinenMultiHeadRMSNorm, torch_module: TorchMultiHeadRMSNorm, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and PyTorch MultiHeadRMSNorm modules.

    Args:
        linen_module: Linen MultiHeadRMSNorm module
        torch_module: PyTorch MultiHeadRMSNorm module
        reverse: If True, convert from PyTorch to Linen. If False, convert from Linen to PyTorch.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # PyTorch to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Create norm parameters
        norm_params = {}
        if torch_module.scale is not None:
            norm_params["scale"] = jnp.array(torch_module.scale.detach().cpu().numpy())

        params["norm"] = norm_params
        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to PyTorch
        with torch.no_grad():
            if variables:
                if "norm" in variables["params"]:
                    norm_params = variables["params"]["norm"]
                    if "scale" in norm_params and torch_module.scale is not None:
                        torch_module.scale.copy_(_convert_to_torch(np.array(norm_params["scale"])))


@ConversionRegistry.register(LinenLayerNorm, TorchLayerNorm)
def convert_layernorm_linen_torch(
    linen_module: LinenLayerNorm, torch_module: TorchLayerNorm, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and PyTorch LayerNorm modules.

    Args:
        linen_module: Linen LayerNorm module
        torch_module: PyTorch LayerNorm module
        reverse: If True, convert from PyTorch to Linen. If False, convert from Linen to PyTorch.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # PyTorch to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Create norm parameters
        norm_params = {}
        if torch_module.scale is not None:
            norm_params["scale"] = jnp.array(torch_module.scale.detach().cpu().numpy())
        if torch_module.bias is not None:
            norm_params["bias"] = jnp.array(torch_module.bias.detach().cpu().numpy())

        params["norm"] = norm_params
        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to PyTorch
        with torch.no_grad():
            if variables:
                if "norm" in variables["params"]:
                    norm_params = variables["params"]["norm"]
                    if "scale" in norm_params and torch_module.scale is not None:
                        torch_module.scale.copy_(_convert_to_torch(np.array(norm_params["scale"])))
                    if "bias" in norm_params and torch_module.bias is not None:
                        torch_module.bias.copy_(_convert_to_torch(np.array(norm_params["bias"])))


@ConversionRegistry.register(LinenRMSNorm, TorchRMSNorm)
def convert_rmsnorm_linen_torch(
    linen_module: LinenRMSNorm, torch_module: TorchRMSNorm, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and PyTorch RMSNorm modules.

    Args:
        linen_module: Linen RMSNorm module
        torch_module: PyTorch RMSNorm module
        reverse: If True, convert from PyTorch to Linen. If False, convert from Linen to PyTorch.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # PyTorch to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Create norm parameters
        norm_params = {}
        if torch_module.scale is not None:
            norm_params["scale"] = jnp.array(torch_module.scale.detach().cpu().numpy())

        params["norm"] = norm_params
        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to PyTorch
        with torch.no_grad():
            if variables:
                if "norm" in variables["params"]:
                    norm_params = variables["params"]["norm"]
                    if "scale" in norm_params and torch_module.scale is not None:
                        torch_module.scale.copy_(_convert_to_torch(np.array(norm_params["scale"])))


@ConversionRegistry.register(LinenIdentity, TorchIdentity)
def convert_identity_linen_torch(
    linen_module: LinenIdentity, torch_module: TorchIdentity, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and PyTorch Identity modules.

    This is a no-op since Identity has no parameters, but included for completeness.

    Args:
        linen_module: Linen Identity module
        torch_module: PyTorch Identity module
        reverse: If True, convert from PyTorch to Linen. If False, convert from Linen to PyTorch.
        **kwargs: Additional arguments including variables for Linen
    """
    # Identity has no parameters, so this is a no-op
    if reverse:
        # PyTorch to Linen
        return kwargs.get("variables", {}).copy()


# Linen <-> NNX conversions


@ConversionRegistry.register(LinenMultiHeadLayerNorm, NNXMultiHeadLayerNorm)
def convert_multihead_layernorm_linen_nnx(
    linen_module: LinenMultiHeadLayerNorm, nnx_module: NNXMultiHeadLayerNorm, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and NNX MultiHeadLayerNorm modules.

    Args:
        linen_module: Linen MultiHeadLayerNorm module
        nnx_module: NNX MultiHeadLayerNorm module
        reverse: If True, convert from NNX to Linen. If False, convert from Linen to NNX.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # NNX to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Create norm parameters
        norm_params = {}
        if (
            hasattr(nnx_module.norm, "scale")
            and nnx_module.norm.scale is not None
            and nnx_module.norm.scale.value is not None
        ):
            norm_params["scale"] = jnp.array(nnx_module.norm.scale.value)
        if (
            hasattr(nnx_module.norm, "bias")
            and nnx_module.norm.bias is not None
            and nnx_module.norm.bias.value is not None
        ):
            norm_params["bias"] = jnp.array(nnx_module.norm.bias.value)

        params["norm"] = norm_params
        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to NNX
        if variables:
            if "norm" in variables["params"]:
                norm_params = variables["params"]["norm"]
                if "scale" in norm_params:
                    nnx_module.norm.scale = nnx.Param(jnp.array(norm_params["scale"]))
                if "bias" in norm_params:
                    nnx_module.norm.bias = nnx.Param(jnp.array(norm_params["bias"]))


@ConversionRegistry.register(LinenMultiHeadRMSNorm, NNXMultiHeadRMSNorm)
def convert_multihead_rmsnorm_linen_nnx(
    linen_module: LinenMultiHeadRMSNorm, nnx_module: NNXMultiHeadRMSNorm, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and NNX MultiHeadRMSNorm modules.

    Args:
        linen_module: Linen MultiHeadRMSNorm module
        nnx_module: NNX MultiHeadRMSNorm module
        reverse: If True, convert from NNX to Linen. If False, convert from Linen to NNX.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # NNX to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Create norm parameters
        norm_params = {}
        if (
            hasattr(nnx_module.norm, "scale")
            and nnx_module.norm.scale is not None
            and nnx_module.norm.scale.value is not None
        ):
            norm_params["scale"] = jnp.array(nnx_module.norm.scale.value)

        params["norm"] = norm_params
        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to NNX
        if variables:
            if "norm" in variables["params"]:
                norm_params = variables["params"]["norm"]
                if "scale" in norm_params:
                    nnx_module.norm.scale = nnx.Param(jnp.array(norm_params["scale"]))


@ConversionRegistry.register(LinenLayerNorm, NNXLayerNorm)
def convert_layernorm_linen_nnx(
    linen_module: LinenLayerNorm, nnx_module: NNXLayerNorm, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and NNX LayerNorm modules.

    Args:
        linen_module: Linen LayerNorm module
        nnx_module: NNX LayerNorm module
        reverse: If True, convert from NNX to Linen. If False, convert from Linen to NNX.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # NNX to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Create norm parameters
        norm_params = {}
        if (
            hasattr(nnx_module.norm, "scale")
            and nnx_module.norm.scale is not None
            and nnx_module.norm.scale.value is not None
        ):
            norm_params["scale"] = jnp.array(nnx_module.norm.scale.value)
        if (
            hasattr(nnx_module.norm, "bias")
            and nnx_module.norm.bias is not None
            and nnx_module.norm.bias.value is not None
        ):
            norm_params["bias"] = jnp.array(nnx_module.norm.bias.value)

        params["norm"] = norm_params
        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to NNX
        if variables:
            if "norm" in variables["params"]:
                norm_params = variables["params"]["norm"]
                if "scale" in norm_params:
                    nnx_module.norm.scale = nnx.Param(jnp.array(norm_params["scale"]))
                if "bias" in norm_params:
                    nnx_module.norm.bias = nnx.Param(jnp.array(norm_params["bias"]))


@ConversionRegistry.register(LinenRMSNorm, NNXRMSNorm)
def convert_rmsnorm_linen_nnx(linen_module: LinenRMSNorm, nnx_module: NNXRMSNorm, *, reverse: bool = False, **kwargs):
    """Convert parameters between Linen and NNX RMSNorm modules.

    Args:
        linen_module: Linen RMSNorm module
        nnx_module: NNX RMSNorm module
        reverse: If True, convert from NNX to Linen. If False, convert from Linen to NNX.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:
        # NNX to Linen
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Create norm parameters
        norm_params = {}
        if (
            hasattr(nnx_module.norm, "scale")
            and nnx_module.norm.scale is not None
            and nnx_module.norm.scale.value is not None
        ):
            norm_params["scale"] = jnp.array(nnx_module.norm.scale.value)

        params["norm"] = norm_params
        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to NNX
        if variables:
            if "norm" in variables["params"]:
                norm_params = variables["params"]["norm"]
                if "scale" in norm_params:
                    nnx_module.norm.scale = nnx.Param(jnp.array(norm_params["scale"]))


@ConversionRegistry.register(LinenIdentity, NNXIdentity)
def convert_identity_linen_nnx(
    linen_module: LinenIdentity, nnx_module: NNXIdentity, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and NNX Identity modules.

    This is a no-op since Identity has no parameters, but included for completeness.

    Args:
        linen_module: Linen Identity module
        nnx_module: NNX Identity module
        reverse: If True, convert from NNX to Linen. If False, convert from Linen to NNX.
        **kwargs: Additional arguments including variables for Linen
    """
    # Identity has no parameters, so this is a no-op
    if reverse:
        # NNX to Linen
        return kwargs.get("variables", {}).copy()
