import jax.numpy as jnp
import jax
from flax import linen as nn

from plstm.config.source_mark_layer import (
    SourceLayerConfig,
    MarkLayerConfig,
    DirectLayerConfig,
)
from .dtype import str_dtype_to_jax
from .initialization import InitInterface


class SourceLayer(nn.Module):
    config: SourceLayerConfig

    def setup(self):
        # Initialize bias
        bias_shape = [self.config.num_heads, self.config.JT, self.config.JK, self.config.JV]
        param_dtype = str_dtype_to_jax(self.config.param_dtype)

        self.bias = self.param(
            "bias",
            self.config.bias_init.instantiate(InitInterface),
            bias_shape,
            param_dtype,
        )

        # Initialize weight if needed
        if self.config.weight:
            weight_shape = [
                self.config.num_heads // self.config.sub_heads,
                self.config.sub_heads,
                self.config.JT,
                self.config.JK,
                self.config.JV,
                self.config.input_dim // self.config.sub_heads,
            ]
            self.weight = self.param(
                "weight",
                self.config.weight_init.instantiate(InitInterface),
                weight_shape,
                param_dtype,
            )

    def _activation(self, x):
        if self.config.activation == "identity":
            return x
        elif self.config.activation == "tanh":
            return self.config.activation_scale * jnp.tanh(x / self.config.activation_scale)
        elif self.config.activation == "logsigmoid":
            return jnp.exp(self.config.activation_scale * jax.nn.log_sigmoid(x))
        else:
            raise ValueError("Bad source activation")

    def __call__(self, x: jax.Array) -> jax.Array:
        # Cast to computation dtype
        dtype = str_dtype_to_jax(self.config.dtype)
        bias = self.bias.astype(dtype)
        x = x.astype(dtype)
        x = x.reshape(*x.shape[:-1], self.config.sub_heads, self.config.input_dim // self.config.sub_heads)

        if self.config.weight:
            weight = self.weight.astype(dtype)
            return self._activation(
                bias
                + jnp.einsum("hsijkd,...sd->...hsijk", weight, x).reshape(
                    *x.shape[:-2], self.config.num_heads, self.config.JT, self.config.JK, self.config.JV
                )
            )
        else:
            return self._activation(
                jnp.tile(bias.reshape(*((1,) * (x.ndim - 2)), *bias.shape), x.shape[:-2] + (1, 1, 1, 1))
            )


class MarkLayer(nn.Module):
    config: MarkLayerConfig

    def setup(self):
        # Initialize bias
        bias_shape = [self.config.num_heads, self.config.JO, self.config.JQ, self.config.JT]
        param_dtype = str_dtype_to_jax(self.config.param_dtype)

        self.bias = self.param(
            "bias",
            self.config.bias_init.instantiate(InitInterface),
            bias_shape,
            param_dtype,
        )

        # Initialize weight if needed
        if self.config.weight:
            weight_shape = [
                self.config.num_heads // self.config.sub_heads,
                self.config.sub_heads,
                self.config.JO,
                self.config.JQ,
                self.config.JT,
                self.config.input_dim // self.config.sub_heads,
            ]
            self.weight = self.param(
                "weight",
                self.config.weight_init.instantiate(InitInterface),
                weight_shape,
                param_dtype,
            )

    def _activation(self, x):
        if self.config.activation == "identity":
            return x
        elif self.config.activation == "tanh":
            return self.config.activation_scale * jnp.tanh(x / self.config.activation_scale)
        elif self.config.activation == "logsigmoid":
            return jnp.exp(self.config.activation_scale * jax.nn.log_sigmoid(x))
        else:
            raise ValueError("Bad mark activation")

    def __call__(self, x: jax.Array) -> jax.Array:
        # Cast to computation dtype
        dtype = str_dtype_to_jax(self.config.dtype)
        bias = self.bias.astype(dtype)
        x = x.astype(dtype)
        x = x.reshape(*x.shape[:-1], self.config.sub_heads, self.config.input_dim // self.config.sub_heads)

        if self.config.weight:
            weight = self.weight.astype(dtype)
            return self._activation(
                bias
                + jnp.einsum("hsijkd,...sd->...hsijk", weight, x).reshape(
                    *x.shape[:-2], self.config.num_heads, self.config.JO, self.config.JQ, self.config.JT
                )
            )
        else:
            return self._activation(
                jnp.tile(bias.reshape(*((1,) * (x.ndim - 2)), *bias.shape), x.shape[:-2] + (1, 1, 1, 1))
            )


class DirectLayer(nn.Module):
    config: DirectLayerConfig

    def setup(self):
        # Initialize bias
        bias_shape = [self.config.num_heads, self.config.JO, self.config.JQ, self.config.JK, self.config.JV]
        param_dtype = str_dtype_to_jax(self.config.param_dtype)

        self.bias = self.param(
            "bias",
            self.config.bias_init.instantiate(InitInterface),
            bias_shape,
            param_dtype,
        )

        # Initialize weight if needed
        if self.config.weight:
            weight_shape = [
                self.config.num_heads // self.config.sub_heads,
                self.config.sub_heads,
                self.config.JO,
                self.config.JQ,
                self.config.JK,
                self.config.JV,
                self.config.input_dim // self.config.sub_heads,
            ]
            self.weight = self.param(
                "weight",
                self.config.weight_init.instantiate(InitInterface),
                weight_shape,
                param_dtype,
            )

    def _activation(self, x):
        if self.config.activation == "identity":
            return x
        elif self.config.activation == "tanh":
            return self.config.activation_scale * jnp.tanh(x / self.config.activation_scale)
        elif self.config.activation == "logsigmoid":
            return jnp.exp(self.config.activation_scale * jax.nn.log_sigmoid(x))
        else:
            raise ValueError("Bad direct activation")

    def __call__(self, x: jax.Array) -> jax.Array:
        # Cast to computation dtype
        dtype = str_dtype_to_jax(self.config.dtype)
        bias = self.bias.astype(dtype)
        x = x.astype(dtype)
        x = x.reshape(*x.shape[:-1], self.config.sub_heads, self.config.input_dim // self.config.sub_heads)

        if self.config.weight:
            weight = self.weight.astype(dtype)
            return self._activation(
                bias
                + jnp.einsum("hsijkld,...sd->...hsijkl", weight, x).reshape(
                    *x.shape[:-2], self.config.num_heads, self.config.JO, self.config.JQ, self.config.JK, self.config.JV
                )
            )
        else:
            return self._activation(
                jnp.tile(bias.reshape(*((1,) * (x.ndim - 2)), *bias.shape), x.shape[:-2] + (1, 1, 1, 1, 1))
            )
