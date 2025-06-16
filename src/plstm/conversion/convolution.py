"""Parameter conversion implementations for convolution layers."""

import torch
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx

from .base import ConversionRegistry, _convert_to_torch
from ..nnx.convolution import Convolution1DLayer as NNXConvolution1DLayer
from ..nnx.convolution import Convolution2DLayer as NNXConvolution2DLayer
from ..torch.convolution import Convolution1DLayer as TorchConvolution1DLayer
from ..torch.convolution import Convolution2DLayer as TorchConvolution2DLayer
from ..linen.convolution import Convolution1DLayer as LinenConvolution1DLayer
from ..linen.convolution import Convolution2DLayer as LinenConvolution2DLayer


# NNX <-> PyTorch conversions


@ConversionRegistry.register(NNXConvolution1DLayer, TorchConvolution1DLayer)
def convert_convolution_1d_nnx_torch(
    nnx_module: NNXConvolution1DLayer, torch_module: TorchConvolution1DLayer, *, reverse: bool = False, **kwargs
):
    """Convert parameters for Convolution1DLayer between NNX and PyTorch.

    The weight tensor needs to be transposed between NNX and PyTorch formats:
    - NNX format: (kernel_size, in_channels, out_channels)
    - PyTorch format: (out_channels, in_channels, kernel_size)

    Args:
        nnx_module: NNX Convolution1DLayer module
        torch_module: PyTorch Convolution1DLayer module
        reverse: If True, convert from PyTorch to NNX. If False, convert from NNX to PyTorch.
        **kwargs: Additional arguments (not used)
    """
    if not reverse:  # NNX to PyTorch
        with torch.no_grad():
            torch_module._conv.weight.copy_(_convert_to_torch(np.array(nnx_module._conv.kernel).transpose(2, 1, 0)))
            if (
                hasattr(nnx_module._conv, "bias")
                and nnx_module._conv.bias is not None
                and nnx_module._conv.bias.value is not None
            ):
                torch_module._conv.bias.copy_(_convert_to_torch(np.array(nnx_module._conv.bias)))
    else:  # PyTorch to NNX
        # Transpose back to NNX format
        kernel = torch_module._conv.weight.detach().cpu().numpy().transpose(2, 1, 0)
        nnx_module._conv.kernel = nnx.Param(jnp.array(kernel))

        if hasattr(torch_module._conv, "bias") and torch_module._conv.bias is not None:
            bias = torch_module._conv.bias.detach().cpu().numpy()
            nnx_module._conv.bias = nnx.Param(jnp.array(bias))


@ConversionRegistry.register(NNXConvolution2DLayer, TorchConvolution2DLayer)
def convert_convolution_2d_nnx_torch(
    nnx_module: NNXConvolution2DLayer, torch_module: TorchConvolution2DLayer, *, reverse: bool = False, **kwargs
):
    """Convert parameters for Convolution2DLayer between NNX and PyTorch.

    The weight tensor needs to be transposed between NNX and PyTorch formats:
    - NNX format: (kernel_height, kernel_width, in_channels, out_channels)
    - PyTorch format: (out_channels, in_channels, kernel_height, kernel_width)

    Args:
        nnx_module: NNX Convolution2DLayer module
        torch_module: PyTorch Convolution2DLayer module
        reverse: If True, convert from PyTorch to NNX. If False, convert from NNX to PyTorch.
        **kwargs: Additional arguments (not used)
    """
    if not reverse:  # NNX to PyTorch
        # Transpose from NNX to PyTorch format
        with torch.no_grad():
            kernel = np.array(nnx_module._conv.kernel).transpose(3, 2, 0, 1)
            torch_module._conv.weight.copy_(_convert_to_torch(kernel))
            if (
                hasattr(nnx_module._conv, "bias")
                and nnx_module._conv.bias is not None
                and nnx_module._conv.bias.value is not None
            ):
                torch_module._conv.bias.copy_(_convert_to_torch(np.array(nnx_module._conv.bias)))
    else:  # PyTorch to NNX
        # Transpose back to NNX format
        kernel = torch_module._conv.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)
        nnx_module._conv.kernel = nnx.Param(jnp.array(kernel))

        if hasattr(torch_module._conv, "bias") and torch_module._conv.bias is not None:
            bias = torch_module._conv.bias.detach().cpu().numpy()
            nnx_module._conv.bias = nnx.Param(jnp.array(bias))


# Linen <-> PyTorch conversions


@ConversionRegistry.register(LinenConvolution1DLayer, TorchConvolution1DLayer)
def convert_convolution_1d_linen_torch(
    linen_module: LinenConvolution1DLayer, torch_module: TorchConvolution1DLayer, *, reverse: bool = False, **kwargs
):
    """Convert parameters for Convolution1DLayer between Linen and PyTorch.

    The weight tensor needs to be transposed between Linen and PyTorch formats:
    - Linen format: (kernel_size, in_channels, out_channels)
    - PyTorch format: (out_channels, in_channels, kernel_size)

    Args:
        linen_module: Linen Convolution1DLayer module
        torch_module: PyTorch Convolution1DLayer module
        reverse: If True, convert from PyTorch to Linen. If False, convert from Linen to PyTorch.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:  # PyTorch to Linen
        # Create updated variables
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Get the kernel parameter and transpose to Linen format
        kernel = torch_module._conv.weight.detach().cpu().numpy().transpose(2, 1, 0)
        params["_conv"] = {"kernel": jnp.array(kernel)}

        # Handle bias if present
        if hasattr(torch_module._conv, "bias") and torch_module._conv.bias is not None:
            bias = torch_module._conv.bias.detach().cpu().numpy()
            params["_conv"]["bias"] = jnp.array(bias)

        updated_variables["params"] = params
        return updated_variables
    else:  # Linen to PyTorch
        with torch.no_grad():
            # Get the kernel parameter from variables and transpose to PyTorch format
            if "_conv" in variables["params"] and "kernel" in variables["params"]["_conv"]:
                kernel = variables["params"]["_conv"]["kernel"]
                torch_module._conv.weight.copy_(_convert_to_torch(np.array(kernel).transpose(2, 1, 0)))

            # Handle bias if present
            if "_conv" in variables["params"] and "bias" in variables["params"]["_conv"]:
                bias = variables["params"]["_conv"]["bias"]
                torch_module._conv.bias.copy_(_convert_to_torch(np.array(bias)))


@ConversionRegistry.register(LinenConvolution2DLayer, TorchConvolution2DLayer)
def convert_convolution_2d_linen_torch(
    linen_module: LinenConvolution2DLayer, torch_module: TorchConvolution2DLayer, *, reverse: bool = False, **kwargs
):
    """Convert parameters for Convolution2DLayer between Linen and PyTorch.

    The weight tensor needs to be transposed between Linen and PyTorch formats:
    - Linen format: (kernel_height, kernel_width, in_channels, out_channels)
    - PyTorch format: (out_channels, in_channels, kernel_height, kernel_width)

    Args:
        linen_module: Linen Convolution2DLayer module
        torch_module: PyTorch Convolution2DLayer module
        reverse: If True, convert from PyTorch to Linen. If False, convert from Linen to PyTorch.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:  # PyTorch to Linen
        # Create updated variables
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Get the kernel parameter and transpose to Linen format
        kernel = torch_module._conv.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)
        params["_conv"] = {"kernel": jnp.array(kernel)}

        # Handle bias if present
        if hasattr(torch_module._conv, "bias") and torch_module._conv.bias is not None:
            bias = torch_module._conv.bias.detach().cpu().numpy()
            params["_conv"]["bias"] = jnp.array(bias)

        updated_variables["params"] = params
        return updated_variables
    else:  # Linen to PyTorch
        with torch.no_grad():
            # Get the kernel parameter from variables and transpose to PyTorch format
            if "_conv" in variables["params"] and "kernel" in variables["params"]["_conv"]:
                kernel = variables["params"]["_conv"]["kernel"]
                torch_module._conv.weight.copy_(_convert_to_torch(np.array(kernel)).transpose(3, 2, 0, 1))

            # Handle bias if present
            if "_conv" in variables["params"] and "bias" in variables["params"]["_conv"]:
                bias = variables["params"]["_conv"]["bias"]
                torch_module._conv.bias.copy_(_convert_to_torch(np.array(bias)))


# Linen <-> NNX conversions


@ConversionRegistry.register(LinenConvolution1DLayer, NNXConvolution1DLayer)
def convert_convolution_1d_linen_nnx(
    linen_module: LinenConvolution1DLayer, nnx_module: NNXConvolution1DLayer, *, reverse: bool = False, **kwargs
):
    """Convert parameters for Convolution1DLayer between Linen and NNX.

    Args:
        linen_module: Linen Convolution1DLayer module
        nnx_module: NNX Convolution1DLayer module
        reverse: If True, convert from NNX to Linen. If False, convert from Linen to NNX.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:  # NNX to Linen
        # Create updated variables
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Get the kernel parameter
        kernel = nnx_module._conv.kernel.value
        params["_conv"] = {"kernel": jnp.array(kernel)}

        # Handle bias if present
        if (
            hasattr(nnx_module._conv, "bias")
            and nnx_module._conv.bias is not None
            and nnx_module._conv.bias.value is not None
        ):
            bias = nnx_module._conv.bias.value
            params["_conv"]["bias"] = jnp.array(bias)

        updated_variables["params"] = params
        return updated_variables
    else:  # Linen to NNX
        # Get the kernel parameter from variables
        if "_conv" in variables["params"] and "kernel" in variables["params"]["_conv"]:
            kernel = variables["params"]["_conv"]["kernel"]
            nnx_module._conv.kernel = nnx.Param(jnp.array(kernel))

        # Handle bias if present
        if "_conv" in variables["params"] and "bias" in variables["params"]["_conv"]:
            bias = variables["params"]["_conv"]["bias"]
            nnx_module._conv.bias = nnx.Param(jnp.array(bias))


@ConversionRegistry.register(LinenConvolution2DLayer, NNXConvolution2DLayer)
def convert_convolution_2d_linen_nnx(
    linen_module: LinenConvolution2DLayer, nnx_module: NNXConvolution2DLayer, *, reverse: bool = False, **kwargs
):
    """Convert parameters for Convolution2DLayer between Linen and NNX.

    Args:
        linen_module: Linen Convolution2DLayer module
        nnx_module: NNX Convolution2DLayer module
        reverse: If True, convert from NNX to Linen. If False, convert from Linen to NNX.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    if reverse:  # NNX to Linen
        # Create updated variables
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Get the kernel parameter
        kernel = nnx_module._conv.kernel.value
        params["_conv"] = {"kernel": jnp.array(kernel)}

        # Handle bias if present
        if (
            hasattr(nnx_module._conv, "bias")
            and nnx_module._conv.bias is not None
            and nnx_module._conv.bias.value is not None
        ):
            bias = nnx_module._conv.bias.value
            params["_conv"]["bias"] = jnp.array(bias)

        updated_variables["params"] = params
        return updated_variables
    else:  # Linen to NNX
        # Get the kernel parameter from variables
        if "_conv" in variables["params"] and "kernel" in variables["params"]["_conv"]:
            kernel = variables["params"]["_conv"]["kernel"]
            nnx_module._conv.kernel = nnx.Param(jnp.array(kernel))

        # Handle bias if present
        if "_conv" in variables["params"] and "bias" in variables["params"]["_conv"]:
            bias = variables["params"]["_conv"]["bias"]
            nnx_module._conv.bias = nnx.Param(jnp.array(bias))
