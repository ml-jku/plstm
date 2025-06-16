from collections.abc import Iterable
from functools import lru_cache
import jax
import jax.numpy as jnp
from functools import wraps
from math import prod
import logging
from plstm.nnx_dummy import nnx

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


# This is just a helper for now, as there seem to be problems with grad ops and nnx.Params
# Interestingly, it is sufficient to just add zero
def weight_einsum(einsum_str: str, x: nnx.Param, y: jax.Array) -> jax.Array:
    return jnp.einsum(einsum_str, x + 0.0, y)


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


def module_named_params(module: nnx.Module, recursive: bool = False) -> Iterable[tuple[str, nnx.Param]]:
    if recursive:
        for submodule_name, submodule in module.iter_children():
            for named_param in module_named_params(submodule, recursive=recursive):
                yield (f"{submodule_name}.{named_param[0]}", named_param[1])

    for param_name in vars(module):
        potential_par = getattr(module, param_name)
        if isinstance(potential_par, nnx.Param) and potential_par.value is not None:
            yield param_name, potential_par


def count_parameters(module: nnx.Module) -> int:
    num_pars = 0
    for _, par in module_named_params(module=module, recursive=True):
        num_pars += par.size

    return num_pars


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
