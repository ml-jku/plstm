import torch
from plstm.nnx_dummy import nnx
import jax.numpy as jnp
import numpy as np
from ..nnx.util import module_named_params as nnx_module_named_params, count_parameters as count_parameters_nnx
from ..torch.util import count_parameters as count_parameters_torch
from .base import (
    ConversionRegistry,
    _convert_to_torch,
    _get_param_names_and_shape_torch,
    _get_param_names_and_shape_nnx,
)


def _default_conversion_nnx_to_torch(
    nnx_module: nnx.Module, torch_module: torch.nn.Module, strict: bool = True
) -> None:
    """Default parameter conversion from NNX to PyTorch."""
    # Get parameters only for this module level
    jax_params = dict(nnx_module_named_params(nnx_module, recursive=False))
    torch_params = dict(torch_module.named_parameters(recurse=False))

    # Copy matching parameters
    for name, jax_param in jax_params.items():
        if name in torch_params:
            with torch.no_grad():
                torch_params[name].copy_(_convert_to_torch(np.array(jax_param)))
        elif strict:
            raise ValueError(f"Parameter {name} not found in torch module")


def _default_conversion_torch_to_nnx(
    torch_module: torch.nn.Module, nnx_module: nnx.Module, strict: bool = True
) -> None:
    """Default parameter conversion from PyTorch to NNX."""
    # Get parameters
    torch_params = dict(torch_module.named_parameters(recurse=False))
    jax_params = dict(nnx_module_named_params(nnx_module, recursive=False))

    # Copy matching parameters
    for name, torch_param in torch_params.items():
        if name in jax_params:
            param_array = torch_param.detach().cpu().numpy()
            setattr(nnx_module, name, nnx.Param(jnp.array(param_array)))
        elif strict:
            raise ValueError(f"Parameter {name} not found in nnx module")


def convert_parameters_nnx_to_torch(nnx_module: nnx.Module, torch_module: torch.nn.Module, strict: bool = True) -> None:
    """Convert parameters from an NNX module to a PyTorch module.

    Args:
        nnx_module: Source NNX module
        torch_module: Target PyTorch module to update
        strict: Whether to raise an error if parameters don't match
    """
    assert count_parameters_nnx(nnx_module) == count_parameters_torch(torch_module), (
        f"Different number of parameters: {_get_param_names_and_shape_nnx(nnx_module)} != "
        f"{_get_param_names_and_shape_torch(torch_module)}"
    )

    # Get conversion function for this module type if registered
    conversion_fn = ConversionRegistry.get_conversion(type(nnx_module), type(torch_module))

    # Convert parameters at this level
    if conversion_fn is not None:
        # Use registered conversion
        conversion_fn(nnx_module, torch_module, reverse=False)
    else:
        _default_conversion_nnx_to_torch(nnx_module, torch_module, strict=strict)
        # Handle child modules
        torch_children = dict(torch_module.named_children())
        for name, child in torch_children.items():
            if hasattr(nnx_module, name) and isinstance(getattr(nnx_module, name), nnx.Module):
                convert_parameters_nnx_to_torch(getattr(nnx_module, name), child, strict=strict)


def convert_parameters_torch_to_nnx(torch_module: torch.nn.Module, nnx_module: nnx.Module, strict: bool = True) -> None:
    """Convert parameters from a PyTorch module to an NNX module.

    Args:
        torch_module: Source PyTorch module
        nnx_module: Target NNX module to update
        strict: Whether to raise an error if parameters don't match
    """
    assert count_parameters_nnx(nnx_module) == count_parameters_torch(torch_module), (
        f"Different number of parameters: {_get_param_names_and_shape_nnx(nnx_module)} != "
        f"{_get_param_names_and_shape_torch(torch_module)}"
    )

    # Get conversion function for this module type if registered
    conversion_fn = ConversionRegistry.get_conversion(type(nnx_module), type(torch_module))

    # Convert parameters
    if conversion_fn is not None:
        # Use registered conversion
        conversion_fn(nnx_module, torch_module, reverse=True)
    else:
        _default_conversion_torch_to_nnx(torch_module, nnx_module, strict=strict)
        # Handle child modules
        torch_children = dict(torch_module.named_children())
        for name, child in torch_children.items():
            if hasattr(nnx_module, name) and isinstance(getattr(nnx_module, name), nnx.Module):
                convert_parameters_torch_to_nnx(child, getattr(nnx_module, name), strict=strict)


# NNX <-> PyTorch conversions
@ConversionRegistry.register(nnx.Linear, torch.nn.Linear)
def convert_linear_nnx_torch(nnx_module: nnx.Linear, torch_module: torch.nn.Linear, *, reverse: bool = False, **kwargs):
    """Convert parameters between NNX Linear and PyTorch Linear modules."""
    if reverse:
        # PyTorch to NNX
        nnx_module.kernel = nnx.Param(jnp.array(torch_module.weight.cpu().detach().numpy()).transpose(1, 0))
        if hasattr(torch_module, "bias") and torch_module.bias is not None:
            nnx_module.bias = nnx.Param(jnp.array(torch_module.bias.cpu().detach().numpy()))
    else:
        # NNX to PyTorch
        with torch.no_grad():
            torch_module.weight.data = _convert_to_torch(np.array(nnx_module.kernel)).transpose(1, 0)
            if nnx_module.bias is not None and nnx_module.bias.value is not None:
                torch_module.bias.data = _convert_to_torch(np.array(nnx_module.bias))


@ConversionRegistry.register(nnx.Embed, torch.nn.Embedding)
def convert_embedding_nnx_torch(
    nnx_module: nnx.Embed, torch_module: torch.nn.Embedding, *, reverse: bool = False, **kwargs
):
    """Convert parameters between NNX Embed and PyTorch Embedding modules."""
    if reverse:
        # PyTorch to NNX
        if hasattr(torch_module, "weight") and torch_module.weight is not None:
            nnx_module.embedding = nnx.Param(jnp.array(torch_module.weight.cpu().detach().numpy()))
    else:
        # NNX to PyTorch
        with torch.no_grad():
            if (
                hasattr(nnx_module, "embedding")
                and nnx_module.embedding is not None
                and nnx_module.embedding.value is not None
            ):
                torch_module.weight.data = _convert_to_torch(np.array(nnx_module.embedding))


@ConversionRegistry.register(nnx.LayerNorm, torch.nn.LayerNorm)
def convert_layernorm_nnx_torch(
    nnx_module: nnx.LayerNorm, torch_module: torch.nn.LayerNorm, *, reverse: bool = False, **kwargs
):
    """Convert parameters between NNX LayerNorm and PyTorch LayerNorm
    modules."""
    if reverse:
        # PyTorch to NNX
        if hasattr(torch_module, "weight") and torch_module.weight is not None:
            nnx_module.scale = nnx.Param(jnp.array(torch_module.weight.cpu().detach().numpy()))
        if hasattr(torch_module, "bias") and torch_module.bias is not None:
            nnx_module.bias = nnx.Param(jnp.array(torch_module.bias.cpu().detach().numpy()))
    else:
        # NNX to PyTorch
        with torch.no_grad():
            if hasattr(nnx_module, "scale") and nnx_module.scale is not None and nnx_module.scale.value is not None:
                torch_module.weight.data = _convert_to_torch(np.array(nnx_module.scale))
            if hasattr(nnx_module, "bias") and nnx_module.bias is not None and nnx_module.bias.value is not None:
                torch_module.bias.data = _convert_to_torch(np.array(nnx_module.bias))


@ConversionRegistry.register(nnx.Conv, torch.nn.Conv1d)
def convert_convolution1d_nnx_torch(
    nnx_module: nnx.Conv, torch_module: torch.nn.Conv1d, *, reverse: bool = False, **kwargs
):
    if reverse:
        if hasattr(torch_module, "weight") and torch_module.weight is not None:
            nnx_module.kernel = nnx.Param(jnp.array(torch_module.weight.cpu().detach().numpy()).transpose(2, 1, 0))
        if hasattr(torch_module, "bias") and torch_module.bias is not None:
            nnx_module.bias = nnx.Param(jnp.array(torch_module.weight.cpu().detach().numpy()))
    else:
        with torch.no_grad():
            if hasattr(nnx_module, "kernel") and nnx_module.kernel is not None:
                torch_module.weight.data = _convert_to_torch(np.array(nnx_module.kernel).transpose(2, 1, 0))
            if hasattr(nnx_module, "bias") and nnx_module.bias is not None:
                torch_module.bias.data = _convert_to_torch(np.array(nnx_module.bias))


@ConversionRegistry.register(nnx.Conv, torch.nn.Conv2d)
def convert_convolution2d_nnx_torch(
    nnx_module: nnx.Conv, torch_module: torch.nn.Conv2d, *, reverse: bool = False, **kwargs
):
    if reverse:
        if hasattr(torch_module, "weight") and torch_module.weight is not None:
            nnx_module.kernel = nnx.Param(jnp.array(torch_module.weight.cpu().detach().numpy()).transpose(2, 3, 1, 0))
        if hasattr(torch_module, "bias") and torch_module.bias is not None:
            nnx_module.bias = nnx.Param(jnp.array(torch_module.weight.cpu().detach().numpy()))
    else:
        with torch.no_grad():
            if hasattr(nnx_module, "kernel") and nnx_module.kernel is not None:
                torch_module.weight.data = _convert_to_torch(np.array(nnx_module.kernel).transpose(3, 2, 0, 1))
            if hasattr(nnx_module, "bias") and nnx_module.bias is not None:
                torch_module.bias.data = _convert_to_torch(np.array(nnx_module.bias))
