from compoconf import register
import jax
import jax.numpy as jnp
from plstm.nnx_dummy import nnx
from ..config.blocks import PreUpProjectionBlockConfig, PostUpProjectionBlockConfig
from .norm import NormInterface
from .interfaces import ResidualModule
from .dtype import str_dtype_to_jax
from .initialization import InitInterface


@register
class PreUpProjectionBlock(ResidualModule):
    config: PreUpProjectionBlockConfig

    def __init__(self, config: PreUpProjectionBlockConfig, rngs: nnx.Rngs):
        ResidualModule.__init__(self, config)
        self.config = config

        self.norm = config.norm.instantiate(NormInterface, rngs=rngs)

        self.upproj = nnx.Linear(
            self.config.input_dim,
            2 * self.config.inner_input_dim if self.config.gated else self.config.inner_input_dim,
            rngs=rngs,
            use_bias=config.bias,
            kernel_init=config.upproj_weight_init.instantiate(InitInterface),
            bias_init=config.upproj_bias_init.instantiate(InitInterface),
            dtype=str_dtype_to_jax(self.config.dtype),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
        )

        self.interaction_module = self.config.interaction_module.instantiate(ResidualModule, rngs=rngs)
        """Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py."""
        self.downproj = nnx.Linear(
            self.config.inner_input_dim,
            self.config.input_dim,
            rngs=rngs,
            use_bias=config.bias,
            kernel_init=config.downproj_weight_init.instantiate(InitInterface),
            bias_init=config.downproj_bias_init.instantiate(InitInterface),
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

    def __call__(self, x, deterministic: bool = False):
        # Cast input to computation dtype
        dtype = str_dtype_to_jax(self.config.dtype)
        x = x.astype(dtype)

        x1 = self.norm(x)
        x2 = self.upproj(x1)

        if self.config.gated:
            x3, x4 = jnp.split(x2, [self.config.inner_input_dim], axis=-1)
        else:
            x3 = x2
        y = self.interaction_module(x3)

        if self.config.gated:
            y2 = y.reshape(*x4.shape) * self._gating_function(x4)
        else:
            y2 = y.reshape(*x3.shape)

        y3 = self.downproj(y2)

        if self.config.skip:
            y4 = y3 + x
        else:
            y4 = y3

        return y4


@register
class PostUpProjectionBlock(ResidualModule):
    config: PostUpProjectionBlockConfig

    def __init__(self, config: PostUpProjectionBlockConfig, rngs: nnx.Rngs):
        ResidualModule.__init__(self, config)
        self.config = config

        self.norm = config.norm.instantiate(NormInterface, rngs=rngs)
        self.norm2 = config.norm.instantiate(NormInterface, rngs=rngs)

        self.interaction_module = self.config.interaction_module.instantiate(ResidualModule, rngs=rngs)

        self.drop_path = nnx.Dropout(self.config.drop_path_rate, rngs=rngs)

        if self.config.use_scale:
            self.scale1 = self.config.scale.instantiate(ResidualModule, rngs=rngs)
            self.scale2 = self.config.scale.instantiate(ResidualModule, rngs=rngs)

        self.upproj = nnx.Linear(
            self.config.input_dim,
            2 * self.config.inner_input_dim if self.config.gated else self.config.inner_input_dim,
            rngs=rngs,
            use_bias=config.bias,
            kernel_init=config.upproj_weight_init.instantiate(InitInterface),
            bias_init=config.upproj_bias_init.instantiate(InitInterface),
            dtype=str_dtype_to_jax(self.config.dtype),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
        )

        self.downproj = nnx.Linear(
            self.config.inner_input_dim,
            self.config.input_dim,
            rngs=rngs,
            use_bias=config.bias,
            kernel_init=config.upproj_weight_init.instantiate(InitInterface),
            bias_init=config.upproj_bias_init.instantiate(InitInterface),
            dtype=str_dtype_to_jax(self.config.dtype),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
        )

    def _gating_function(self, x):
        if self.config.gating_function == "silu":
            return jax.nn.silu(x)
        if self.config.gating_function == "gelu":
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
        # Cast input to computation dtype
        dtype = str_dtype_to_jax(self.config.dtype)
        x = x.astype(dtype)

        x1 = self.norm(x)
        y1 = self.interaction_module(x1)

        if self.config.skip:
            if self.config.use_scale:
                y1 = self.scale1(y1)
            x2 = self._drop_path(y1, deterministic=deterministic) + x
        else:
            x2 = y1

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

        y3 = self.downproj(y2)

        if self.config.skip:
            if self.config.use_scale:
                y3 = self.scale2(y3)
            y4 = self._drop_path(y3, deterministic=deterministic) + x2
        else:
            y4 = y3

        return y4
