import jax.numpy as jnp
import jax
from plstm.nnx_dummy import nnx
from .util import weight_einsum
from .dtype import str_dtype_to_jax
from .initialization import InitInterface

from plstm.config.query_key_value import (
    QueryLayerConfig,
    KeyLayerConfig,
    ValueLayerConfig,
)


class QueryLayer(nnx.Module):
    config: QueryLayerConfig

    def __init__(self, config: QueryLayerConfig, *, rngs: nnx.Rngs):
        nnx.Module.__init__(self)
        self.config = config

        # Initialize weight
        weight_shape = [
            max(1, config.num_heads // config.sub_heads),
            config.DK // max(1, config.sub_heads // config.num_heads),
            config.JQ,
            config.sub_heads,
            config.input_dim // config.sub_heads,
        ]

        self.weight = nnx.Param(
            config.weight_init.instantiate(InitInterface)(
                rngs.params(), weight_shape, dtype=str_dtype_to_jax(config.param_dtype)
            )
        )

        # Initialize bias if needed
        if self.config.bias:
            bias_shape = [config.num_heads, config.DK, config.JQ]
            self.bias = nnx.Param(
                config.bias_init.instantiate(InitInterface)(
                    rngs.params(), bias_shape, dtype=str_dtype_to_jax(config.param_dtype)
                )
            )

    def _normalization(self, x):
        if self.config.normalization == "none":
            return x
        elif self.config.normalization == "l2":
            x = x.astype(str_dtype_to_jax(self.config.dtype))
            return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5)

    def __call__(self, x) -> jax.Array:
        x_reshaped = x.reshape(
            *x.shape[:-1],
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        )
        # Cast to computation dtype for mixed precision
        weight = self.weight.astype(str_dtype_to_jax(self.config.dtype))
        x_reshaped = x_reshaped.astype(str_dtype_to_jax(self.config.dtype))
        out = weight_einsum("hkjsx,...sx->...hskj", weight, x_reshaped)
        out = out.reshape(list(x.shape[:-1]) + [self.config.num_heads, self.config.DK, self.config.JQ])
        if self.config.bias:
            bias = self.bias.astype(str_dtype_to_jax(self.config.dtype))
            out = out + bias
        return self._normalization(out)

    def __repr__(self):
        return (
            f"QueryLayer(heads={self.config.num_heads}, subheads={self.config.sub_heads},"
            f" input_dim={self.config.input_dim}, JQ={self.config.JQ}, DK={self.config.DK})"
        )


class KeyLayer(nnx.Module):
    config: KeyLayerConfig

    def __init__(self, config: KeyLayerConfig, *, rngs: nnx.Rngs):
        nnx.Module.__init__(self)
        self.config = config

        # Initialize weight
        weight_shape = [
            max(1, config.num_heads // config.sub_heads),
            config.DK // max(1, config.sub_heads // config.num_heads),
            config.JK,
            config.sub_heads,
            config.input_dim // config.sub_heads,
        ]

        self.weight = nnx.Param(
            config.weight_init.instantiate(InitInterface)(
                rngs.params(), weight_shape, dtype=str_dtype_to_jax(config.param_dtype)
            )
        )

        # Initialize bias if needed
        if self.config.bias:
            bias_shape = [config.num_heads, config.DK, config.JK]
            self.bias = nnx.Param(
                config.bias_init.instantiate(InitInterface)(
                    rngs.params(), bias_shape, dtype=str_dtype_to_jax(config.param_dtype)
                )
            )

    def _normalization(self, x):
        if self.config.normalization == "none":
            return x
        elif self.config.normalization == "l2":
            x = x.astype(str_dtype_to_jax(self.config.dtype))
            return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5)

    def __call__(self, x) -> jax.Array:
        x_reshaped = x.reshape(
            *x.shape[:-1],
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        )
        # Cast to computation dtype for mixed precision
        weight = self.weight.astype(str_dtype_to_jax(self.config.dtype))
        x_reshaped = x_reshaped.astype(str_dtype_to_jax(self.config.dtype))
        out = weight_einsum("hkjsx,...sx->...hskj", weight, x_reshaped)
        out = out.reshape(list(x.shape[:-1]) + [self.config.num_heads, self.config.DK, self.config.JK])
        if self.config.bias:
            bias = self.bias.astype(str_dtype_to_jax(self.config.dtype))
            out = out + bias
        return self._normalization(out)

    def __repr__(self):
        return (
            f"KeyLayer(heads={self.config.num_heads}, subheads={self.config.sub_heads},"
            f" input_dim={self.config.input_dim}, JK={self.config.JK}, DK={self.config.DK})"
        )


class ValueLayer(nnx.Module):
    config: ValueLayerConfig

    def __init__(self, config: ValueLayerConfig, *, rngs: nnx.Rngs):
        nnx.Module.__init__(self)
        self.config = config

        # Initialize weight
        weight_shape = [
            max(1, config.num_heads // config.sub_heads),
            config.DV // max(1, config.sub_heads // config.num_heads),
            config.JV,
            config.sub_heads,
            config.input_dim // config.sub_heads,
        ]

        self.weight = nnx.Param(
            config.weight_init.instantiate(InitInterface)(
                rngs.params(), weight_shape, dtype=str_dtype_to_jax(config.param_dtype)
            )
        )

        # Initialize bias if needed
        if self.config.bias:
            bias_shape = [config.num_heads, config.DV, config.JV]
            self.bias = nnx.Param(
                config.bias_init.instantiate(InitInterface)(
                    rngs.params(), bias_shape, dtype=str_dtype_to_jax(config.param_dtype)
                )
            )

    def _normalization(self, x):
        if self.config.normalization == "none":
            return x
        elif self.config.normalization == "l2":
            x = x.astype(str_dtype_to_jax(self.config.dtype))
            return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5)

    def __call__(self, x) -> jax.Array:
        x_reshaped = x.reshape(
            *x.shape[:-1],
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        )
        # Cast to computation dtype for mixed precision
        weight = self.weight.astype(str_dtype_to_jax(self.config.dtype))
        x_reshaped = x_reshaped.astype(str_dtype_to_jax(self.config.dtype))
        out = weight_einsum("hkjsx,...sx->...hskj", weight, x_reshaped)
        out = out.reshape(list(x.shape[:-1]) + [self.config.num_heads, self.config.DV, self.config.JV])
        if self.config.bias:
            bias = self.bias.astype(str_dtype_to_jax(self.config.dtype))
            out = out + bias
        return self._normalization(out)

    def __repr__(self):
        return (
            f"ValueLayer(heads={self.config.num_heads}, subheads={self.config.sub_heads},"
            f" input_dim={self.config.input_dim}, JV={self.config.JV}, DV={self.config.DV})"
        )
