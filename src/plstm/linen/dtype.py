from ..config.dtype import DTYPES
import jax.numpy as jnp


def str_dtype_to_jax(dtype: DTYPES) -> jnp.dtype:
    return getattr(jnp, dtype)
