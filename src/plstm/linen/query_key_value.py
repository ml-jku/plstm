import jax.numpy as jnp
import jax
from flax import linen as nn

from plstm.config.query_key_value import (
    QueryLayerConfig,
    KeyLayerConfig,
    ValueLayerConfig,
)
from .dtype import str_dtype_to_jax
from .initialization import InitInterface


class QueryLayer(nn.Module):
    config: QueryLayerConfig

    def _normalization(self, x):
        if self.config.normalization == "none":
            return x
        elif self.config.normalization == "l2":
            x = x.astype(str_dtype_to_jax(self.config.dtype))
            return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5)

    def setup(self):
        # Initialize weight
        weight_shape = [
            max(1, self.config.num_heads // self.config.sub_heads),
            self.config.DK // max(1, self.config.sub_heads // self.config.num_heads),
            self.config.JQ,
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        ]

        self.weight = self.param(
            "weight",
            self.config.weight_init.instantiate(InitInterface),
            weight_shape,
            str_dtype_to_jax(self.config.param_dtype),
        )

        # Initialize bias if needed
        if self.config.bias:
            bias_shape = [self.config.num_heads, self.config.DK, self.config.JQ]
            self.bias = self.param(
                "bias",
                self.config.bias_init.instantiate(InitInterface),
                bias_shape,
                str_dtype_to_jax(self.config.param_dtype),
            )

    def __call__(self, x) -> jax.Array:
        x_reshaped = x.reshape(
            *x.shape[:-1],
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        )
        # Cast to computation dtype for mixed precision
        weight = self.weight.astype(str_dtype_to_jax(self.config.dtype))
        x_reshaped = x_reshaped.astype(str_dtype_to_jax(self.config.dtype))
        out = jnp.einsum("hkjsx,...sx->...hskj", weight, x_reshaped)
        out = out.reshape(list(x.shape[:-1]) + [self.config.num_heads, self.config.DK, self.config.JQ])
        if self.config.bias:
            bias = self.bias.astype(str_dtype_to_jax(self.config.dtype))
            out = out + bias
        return self._normalization(out)


class KeyLayer(nn.Module):
    config: KeyLayerConfig

    def _normalization(self, x):
        if self.config.normalization == "none":
            return x
        elif self.config.normalization == "l2":
            x = x.astype(str_dtype_to_jax(self.config.dtype))
            return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5)

    def setup(self):
        # Initialize weight
        weight_shape = [
            max(1, self.config.num_heads // self.config.sub_heads),
            self.config.DK // max(1, self.config.sub_heads // self.config.num_heads),
            self.config.JK,
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        ]

        self.weight = self.param(
            "weight",
            self.config.weight_init.instantiate(InitInterface),
            weight_shape,
            str_dtype_to_jax(self.config.param_dtype),
        )

        # Initialize bias if needed
        if self.config.bias:
            bias_shape = [self.config.num_heads, self.config.DK, self.config.JK]
            self.bias = self.param(
                "bias",
                self.config.bias_init.instantiate(InitInterface),
                bias_shape,
                str_dtype_to_jax(self.config.param_dtype),
            )

    def __call__(self, x) -> jax.Array:
        x_reshaped = x.reshape(
            *x.shape[:-1],
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        )
        # Cast to computation dtype for mixed precision
        weight = self.weight.astype(str_dtype_to_jax(self.config.dtype))
        x_reshaped = x_reshaped.astype(str_dtype_to_jax(self.config.dtype))
        out = jnp.einsum("hkjsx,...sx->...hskj", weight, x_reshaped)
        out = out.reshape(list(x.shape[:-1]) + [self.config.num_heads, self.config.DK, self.config.JK])
        if self.config.bias:
            bias = self.bias.astype(str_dtype_to_jax(self.config.dtype))
            out = out + bias
        return self._normalization(out)


class ValueLayer(nn.Module):
    config: ValueLayerConfig

    def _normalization(self, x):
        if self.config.normalization == "none":
            return x
        elif self.config.normalization == "l2":
            x = x.astype(str_dtype_to_jax(self.config.dtype))
            return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5)

    def setup(self):
        # Initialize weight
        weight_shape = [
            max(1, self.config.num_heads // self.config.sub_heads),
            self.config.DV // max(1, self.config.sub_heads // self.config.num_heads),
            self.config.JV,
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        ]

        self.weight = self.param(
            "weight",
            self.config.weight_init.instantiate(InitInterface),
            weight_shape,
            str_dtype_to_jax(self.config.param_dtype),
        )

        # Initialize bias if needed
        if self.config.bias:
            bias_shape = [self.config.num_heads, self.config.DV, self.config.JV]
            self.bias = self.param(
                "bias",
                self.config.bias_init.instantiate(InitInterface),
                bias_shape,
                str_dtype_to_jax(self.config.param_dtype),
            )

    def __call__(self, x) -> jax.Array:
        x_reshaped = x.reshape(
            *x.shape[:-1],
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        )
        # Cast to computation dtype for mixed precision
        weight = self.weight.astype(str_dtype_to_jax(self.config.dtype))
        x_reshaped = x_reshaped.astype(str_dtype_to_jax(self.config.dtype))
        out = jnp.einsum("hkjsx,...sx->...hskj", weight, x_reshaped)
        out = out.reshape(list(x.shape[:-1]) + [self.config.num_heads, self.config.DV, self.config.JV])
        if self.config.bias:
            bias = self.bias.astype(str_dtype_to_jax(self.config.dtype))
            out = out + bias
        return self._normalization(out)
