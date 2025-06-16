from functools import lru_cache
from collections.abc import Iterable
import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import wraps
from math import prod
import logging

import flax.traverse_util as traverse_util

LOGGER = logging.getLogger(__name__)


def static_jit(**decorator_kwargs):
    """A decorator to JIT-compile a function with static arguments."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # JIT compile the function with the static argument specified.
            return jax.jit(func, **decorator_kwargs)(*args, **kwargs)

        return wrapper

    return decorator


def allclose_verbose(a, b, **kwargs):
    if a.shape != b.shape:
        print(f"Shape mismatch: {a.shape} != {b.shape}")
        LOGGER.warning(f"Shape mismatch: {a.shape} != {b.shape}")
    return jnp.allclose(a, b, **kwargs)


@lru_cache(maxsize=1024)
def log2(a):
    """
    >>> log2(1)
    0
    >>> log2(2)
    1
    >>> log2(5)
    2
    """
    n = 0
    while a > 1:
        a = a >> 1
        n += 1
    return n


def ispow2(a):
    """
    >>> ispow2(1)
    True
    >>> ispow2(2)
    True
    >>> ispow2(3)
    False
    >>> ispow2(8)
    True
    """
    return (1 << log2(a)) == a


def least_significant_bit(x):
    """
    >>> least_significant_bit(1)
    0
    >>> least_significant_bit(2)
    1
    >>> least_significant_bit(3)
    0
    >>> least_significant_bit(8)
    3
    >>> least_significant_bit(6)
    1
    """
    if x < 0:
        raise ValueError("Input must be > 0")
    for i in range(0, log2(x) + 1):
        d = x - ((x >> (i + 1)) << (i + 1))
        if d:
            return i


def rev_cumsum_off(x: jax.Array) -> jax.Array:
    y = jnp.zeros_like(x)
    y = y.at[..., :-1].set(x[..., 1:])
    return jnp.flip(jnp.cumsum(jnp.flip(y, axis=-1), axis=-1), axis=-1)


def cumsum_off(x: jax.Array) -> jax.Array:
    y = jnp.zeros_like(x)
    y = y.at[..., 1:].set(x[..., :-1])
    return y.cumsum(axis=-1)


def rev_cumsum(x):
    return jnp.flip(jnp.cumsum(jnp.flip(x, axis=-1), axis=-1), axis=-1)


def identity(x: jax.Array) -> jax.Array:
    return x


def plus(x: jax.Array, y: jax.Array) -> jax.Array:
    return x + y


def flatten_axes(ar: jax.Array, start_axis: int = 0, end_axis: int = -1):
    """
    >>> import jax.numpy as jnp
    >>> flatten_axes(jnp.zeros([2,3,4,5]), 1, -2).shape
    (2, 12, 5)
    >>> flatten_axes(jnp.zeros([2,3,4,5]), 1, 2).shape
    (2, 12, 5)
    >>>
    """
    shape = ar.shape
    end_axis = len(shape) + end_axis if end_axis < 0 else end_axis
    new_shape = shape[:start_axis] + (prod(shape[start_axis : end_axis + 1]),) + shape[end_axis + 1 :]
    return ar.reshape(new_shape)


def module_get_children(
    module: nn.Module, rngs: jax.random.PRNGKey, exmp_input: jax.Array, recursive: bool = False
) -> Iterable[tuple[str, nn.Module]]:
    for name, mod in module.module_paths(rngs, exmp_input):
        if isinstance(mod, nn.Module):
            yield (name, mod)


def module_named_params(module: nn.Module, variables, recursive: bool = False) -> Iterable[tuple[str, jax.Array]]:
    """Get named parameters from a Flax Linen module.

    Args:
        module: The Linen module
        variables: The variables dict from module.init() or similar
        recursive: Whether to recursively get parameters from submodules

    Returns:
        Iterable of (name, param) tuples
    """
    if variables:
        if recursive:
            flat_params = traverse_util.flatten_dict(variables["params"])
            for param_name, param in flat_params.items():
                # Convert tuple path to dot notation
                name = ".".join([str(p) for p in param_name])
                yield name, param
        else:
            for param_name, param in variables["params"].items():
                if isinstance(param, jnp.ndarray):
                    yield param_name, param


def count_parameters(module: nn.Module, variables) -> int:
    """Count the number of parameters in a Flax Linen module.

    Args:
        module: The Linen module
        variables: The variables dict from module.init() or similar

    Returns:
        Number of parameters
    """
    num_params = 0
    for _, param in module_named_params(module, variables, recursive=True):
        num_params += param.size
    return num_params


def get_param_names_and_shape(module: nn.Module, variables) -> dict[str, tuple]:
    """Get parameter names and shapes from a Flax Linen module.

    Args:
        module: The Linen module
        variables: The variables dict from module.init() or similar

    Returns:
        Dictionary mapping parameter names to shapes
    """
    return {
        param_name: tuple(param.shape) for param_name, param in module_named_params(module, variables, recursive=True)
    }


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
