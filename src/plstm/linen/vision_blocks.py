import jax
from .interfaces import ResidualModule
from compoconf import register
from ..config.vision_blocks import pLSTMVisionBlockConfig1
from flax import linen as nn
import jax.numpy as jnp


@register
class pLSTMVisionBlock1(ResidualModule):
    config: pLSTMVisionBlockConfig1

    def setup(self):
        self.block0 = self.config.block0.instantiate(ResidualModule)
        self.block1 = self.config.block1.instantiate(ResidualModule)
        self.drop_path = nn.Dropout(self.config.drop_path_rate)

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        x1 = self.block0(x, deterministic=deterministic)
        x2 = self.block1(x1, deterministic=deterministic)
        return (
            self.drop_path(jnp.ones((x.shape[0],), dtype=x.dtype), deterministic=deterministic).reshape(
                x.shape[0], *([1] * (x.ndim - 1))
            )
            * (x2 - x)
        ) + x
