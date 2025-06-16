from compoconf import register
import jax
import jax.numpy as jnp
from flax import linen as nn
from ..config.transformer_block import TransformerBlockConfig
from .norm import NormInterface
from .interfaces import ResidualModule
from .scale import ScaleLayer
from .dtype import str_dtype_to_jax
from .initialization import InitInterface
import pkg_resources


@register
class TransformerBlock(ResidualModule):
    config: TransformerBlockConfig

    def setup(self):
        self.norm = self.config.norm.instantiate(NormInterface)
        self.norm2 = self.config.norm.instantiate(NormInterface)
        self.scale = ScaleLayer(self.config.scale)
        self.scale2 = ScaleLayer(self.config.scale)
        self.drop_path = nn.Dropout(self.config.drop_path_rate)

        try:
            self.multiheadattention = nn.MultiHeadDotProductAttention(
                num_heads=self.config.num_heads,
                qkv_features=self.config.input_dim,
                decode=self.config.decode,
                kernel_init=self.config.attn_weight_init.instantiate(InitInterface),
                out_kernel_init=self.config.out_weight_init.instantiate(InitInterface),
                dtype=str_dtype_to_jax(self.config.dtype),
                param_dtype=str_dtype_to_jax(self.config.param_dtype),
                use_bias=self.config.bias,
            )
        except TypeError:
            self.multiheadattention = nn.MultiHeadDotProductAttention(
                num_heads=self.config.num_heads,
                qkv_features=self.config.input_dim,
                decode=self.config.decode,
                kernel_init=self.config.attn_weight_init.instantiate(InitInterface),
                # out_kernel_init=self.config.out_weight_init.instantiate(InitInterface),
                dtype=str_dtype_to_jax(self.config.dtype),
                param_dtype=str_dtype_to_jax(self.config.param_dtype),
                use_bias=self.config.bias,
            )

        self.upproj = nn.Dense(
            features=2 * self.config.inner_input_dim if self.config.gated else self.config.inner_input_dim,
            use_bias=self.config.bias,
            kernel_init=self.config.upproj_weight_init.instantiate(InitInterface),
            bias_init=self.config.upproj_bias_init.instantiate(InitInterface),
            dtype=str_dtype_to_jax(self.config.dtype),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
        )

        self.downproj = nn.Dense(
            features=self.config.input_dim,
            use_bias=self.config.bias,
            kernel_init=self.config.downproj_weight_init.instantiate(InitInterface),
            bias_init=self.config.downproj_bias_init.instantiate(InitInterface),
            dtype=str_dtype_to_jax(self.config.dtype),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
        )

    def _gating_function(self, x):
        if self.config.gating_function == "silu":
            return jax.nn.silu(x)
        elif self.config.gating_function == "gelu":
            return jax.nn.gelu(x)
        else:
            raise ValueError("Bad gating function")

    def _drop_path(self, x, deterministic: bool):
        return (
            self.drop_path(jnp.ones((x.shape[0],), dtype=x.dtype), deterministic=deterministic).reshape(
                x.shape[0], *([1] * (x.ndim - 1))
            )
            * x
        )

    def __call__(self, x, deterministic: bool = False):
        x0 = self.norm(x)
        x_shape = x0.shape
        x1 = x0.reshape(x_shape[0], -1, x_shape[-1])
        if pkg_resources.get_distribution("flax").version < "0.8.0":
            y1 = self.multiheadattention(inputs_q=x1, inputs_kv=x1).reshape(x_shape)
        else:
            y1 = self.multiheadattention(x1).reshape(x_shape)

        if self.config.skip:
            x2 = self._drop_path(self.scale(y1), deterministic=deterministic) + x
        else:
            x2 = self.scale(y1)

        x3 = self.norm2(x2)
        x4 = self.upproj(x3)

        if self.config.gated:
            x5, x6 = jnp.split(x4, [self.config.inner_input_dim], axis=-1)
        else:
            x5 = x4

        if self.config.gated:
            y2 = x5 * self._gating_function(x6)
        else:
            y2 = self._gating_function(x5)

        y3 = self.scale2(self.downproj(y2))

        if self.config.skip:
            y4 = self._drop_path(y3, deterministic=deterministic) + x2
        else:
            y4 = y3

        return y4
