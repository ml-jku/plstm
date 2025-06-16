from plstm.nnx_dummy import nnx
import jax
from .interfaces import ResidualModule
from compoconf import register
from ..config.vision_blocks import pLSTMVisionBlockConfig1
import jax.numpy as jnp


@register
class pLSTMVisionBlock1(ResidualModule):
    config: pLSTMVisionBlockConfig1

    def __init__(self, config: pLSTMVisionBlockConfig1, rngs: nnx.Rngs):
        ResidualModule.__init__(self, config)
        self.config = config
        self.block0 = config.block0.instantiate(ResidualModule, rngs=rngs)
        self.block1 = config.block1.instantiate(ResidualModule, rngs=rngs)
        self.drop_path = nnx.Dropout(self.config.drop_path_rate, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        x1 = self.block0(x, deterministic=deterministic)
        x2 = self.block1(x1, deterministic=deterministic)
        return (
            self.drop_path(jnp.ones((x.shape[0],), dtype=x.dtype), deterministic=deterministic).reshape(
                x.shape[0], *([1] * (x.ndim - 1))
            )
            * (x2 - x)
        ) + x
