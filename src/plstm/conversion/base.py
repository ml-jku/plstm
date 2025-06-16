"""Base parameter conversion utilities between JAX (nnx/linen) and PyTorch
models."""

import enum
from typing import Any, TypeVar
from collections.abc import Callable

import torch
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from plstm.nnx_dummy import nnx

from ..nnx.util import module_named_params as nnx_module_named_params
from ..linen.util import module_named_params as linen_module_named_params


# Type definitions for better type checking
NNXModule = TypeVar("NNXModule", bound=nnx.Module)
LinenModule = TypeVar("LinenModule", bound=nn.Module)
TorchModule = TypeVar("TorchModule", bound=torch.nn.Module)
Variables = dict[str, Any]


class ModuleType(enum.Enum):
    """Enum for module types."""

    NNX = "nnx"
    LINEN = "linen"
    TORCH = "torch"


def detect_module_type(module: Any) -> ModuleType:
    """Detect the type of a module.

    Args:
        module: The module to detect

    Returns:
        ModuleType enum value

    Raises:
        ValueError: If the module type cannot be determined
    """
    if isinstance(module, nnx.Module):
        return ModuleType.NNX
    elif isinstance(module, nn.Module) or (hasattr(module, "__module__") and "flax.linen" in module.__module__):
        return ModuleType.LINEN
    elif isinstance(module, torch.nn.Module):
        return ModuleType.TORCH
    else:
        raise ValueError(f"Unknown module type: {type(module)}")


def _convert_to_torch(ar: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a torch tensor."""
    if ar.dtype == np.dtype("bfloat16"):
        return torch.from_numpy(ar.astype(np.float32)).to(torch.bfloat16)
    else:
        return torch.from_numpy(ar)


def _convert_to_jax(tensor: torch.Tensor) -> jnp.ndarray:
    """Convert a torch tensor to a jax array."""
    return jnp.array(tensor.detach().cpu().numpy())


class ConversionRegistry:
    """Registry for module-specific parameter conversions."""

    # Dictionary mapping (source_type, target_type) to a dictionary of module-specific conversions
    _conversions: dict[tuple[type, type], dict[str, Callable]] = {}

    @classmethod
    def register(cls, source_type: type, target_type: type, name: str | None = None) -> Callable:
        """Decorator to register a conversion function for a module type.

        Args:
            source_type: Source module type (e.g., nnx.Linear, nn.Dense)
            target_type: Target module type (e.g., torch.nn.Linear)
            name: Optional name for the conversion. If None, uses source_type.__name__

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            key = (source_type, target_type)
            if key not in cls._conversions:
                cls._conversions[key] = {}

            conversion_name = name or source_type.__name__
            cls._conversions[key][conversion_name] = func

            key = (target_type, source_type)
            if key not in cls._conversions:
                cls._conversions[key] = {}

            conversion_name = name or target_type.__name__
            cls._conversions[key][conversion_name] = lambda x, y, **kwargs: func(y, x, reverse=True, **kwargs)

            return func

        return decorator

    @classmethod
    def get_conversion(cls, source_type: type, target_type: type, name: str | None = None) -> Callable | None:
        """Get conversion function for a module type if registered.

        Args:
            source_type: Source module type
            target_type: Target module type
            name: Optional name for the conversion. If None, uses source_type.__name__

        Returns:
            Conversion function or None if not registered
        """
        key = (source_type, target_type)
        if key not in cls._conversions:
            return None

        conversion_name = name or source_type.__name__
        return cls._conversions[key].get(conversion_name)


# Helper functions for parameter conversion


def _get_param_names_and_shape_torch(torch_module: torch.nn.Module) -> dict[str, tuple]:
    """Get parameter names and shapes from a PyTorch module."""
    return {par_name: tuple(par.shape) for par_name, par in torch_module.named_parameters(recurse=True)}


def _get_param_names_and_shape_nnx(nnx_module: nnx.Module) -> dict[str, tuple]:
    """Get parameter names and shapes from an NNX module."""
    return {par_name: tuple(par.shape) for par_name, par in nnx_module_named_params(nnx_module, recursive=True)}


def _get_param_names_and_shape_linen(linen_module: nn.Module, variables: dict) -> dict[str, tuple]:
    """Get parameter names and shapes from a Linen module."""
    return {
        param_name: tuple(param.shape)
        for param_name, param in linen_module_named_params(linen_module, variables, recursive=True)
    }
