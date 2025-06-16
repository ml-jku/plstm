import jax.numpy as jnp
import jax
from plstm.nnx_dummy import nnx

from plstm.config.source_mark_layer import (
    SourceLayerConfig,
    MarkLayerConfig,
    DirectLayerConfig,
)
from .util import weight_einsum
from .dtype import str_dtype_to_jax
from .initialization import InitInterface


class SourceLayer(nnx.Module):
    config: SourceLayerConfig

    def __init__(self, config: SourceLayerConfig, *, rngs: nnx.Rngs):
        nnx.Module.__init__(self)
        self.config = config

        # Initialize bias
        bias_shape = [config.num_heads, config.JT, config.JK, config.JV]

        self.bias = nnx.Param(
            config.bias_init.instantiate(InitInterface)(
                rngs.params(), bias_shape, dtype=str_dtype_to_jax(config.param_dtype)
            )
        )

        # Initialize weight if needed
        if self.config.weight:
            weight_shape = [
                config.num_heads // config.sub_heads,
                config.sub_heads,
                config.JT,
                config.JK,
                config.JV,
                config.input_dim // config.sub_heads,
            ]
            self.weight = nnx.Param(
                config.weight_init.instantiate(InitInterface)(
                    rngs.params(), weight_shape, dtype=str_dtype_to_jax(config.param_dtype)
                )
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
                + weight_einsum("hsijkd,...sd->...hsijk", weight, x).reshape(
                    *x.shape[:-2], self.config.num_heads, self.config.JT, self.config.JK, self.config.JV
                )
            )
        else:
            return self._activation(
                jnp.tile(bias.reshape(*((1,) * (x.ndim - 2)), *bias.shape), x.shape[:-2] + (1, 1, 1, 1))
            )


class MarkLayer(nnx.Module):
    config: MarkLayerConfig

    def __init__(self, config: MarkLayerConfig, *, rngs: nnx.Rngs):
        nnx.Module.__init__(self)
        self.config = config

        # Initialize bias
        bias_shape = [config.num_heads, config.JO, config.JQ, config.JT]
        self.bias = nnx.Param(
            config.bias_init.instantiate(InitInterface)(
                rngs.params(), bias_shape, dtype=str_dtype_to_jax(config.param_dtype)
            )
        )

        # Initialize weight if needed
        if self.config.weight:
            weight_shape = [
                config.num_heads // config.sub_heads,
                config.sub_heads,
                config.JO,
                config.JQ,
                config.JT,
                config.input_dim // config.sub_heads,
            ]
            self.weight = nnx.Param(
                config.weight_init.instantiate(InitInterface)(
                    rngs.params(), weight_shape, dtype=str_dtype_to_jax(config.param_dtype)
                )
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
                + weight_einsum("hsijkd,...sd->...hsijk", weight, x).reshape(
                    *x.shape[:-2], self.config.num_heads, self.config.JO, self.config.JQ, self.config.JT
                )
            )
        else:
            return self._activation(
                jnp.tile(bias.reshape(*((1,) * (x.ndim - 2)), *bias.shape), x.shape[:-2] + (1, 1, 1, 1))
            )


class DirectLayer(nnx.Module):
    config: DirectLayerConfig

    def __init__(self, config: DirectLayerConfig, *, rngs: nnx.Rngs):
        nnx.Module.__init__(self)
        self.config = config

        # Initialize bias
        bias_shape = [config.num_heads, config.JO, config.JQ, config.JK, config.JV]
        self.bias = nnx.Param(
            config.bias_init.instantiate(InitInterface)(
                rngs.params(), bias_shape, dtype=str_dtype_to_jax(config.param_dtype)
            )
        )

        # Initialize weight if needed
        if self.config.weight:
            weight_shape = [
                config.num_heads // config.sub_heads,
                config.sub_heads,
                config.JO,
                config.JQ,
                config.JK,
                config.JV,
                config.input_dim // config.sub_heads,
            ]
            self.weight = nnx.Param(
                config.weight_init.instantiate(InitInterface)(
                    rngs.params(), weight_shape, dtype=str_dtype_to_jax(config.param_dtype)
                )
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
                + weight_einsum("hsijkld,...sd->...hsijkl", weight, x).reshape(
                    *x.shape[:-2], self.config.num_heads, self.config.JO, self.config.JQ, self.config.JK, self.config.JV
                )
            )
        else:
            return self._activation(
                jnp.tile(bias.reshape(*((1,) * (x.ndim - 2)), *bias.shape), x.shape[:-2] + (1, 1, 1, 1, 1))
            )
