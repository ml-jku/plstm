import jax
from flax import linen as nn

from ..config.passthrough import PassthroughLayerConfig
from .dtype import str_dtype_to_jax
from .initialization import InitInterface


class PassthroughLayer(nn.Module):
    """Layer that passes through input with learned scale."""

    config: PassthroughLayerConfig

    def setup(self):
        param_dtype = str_dtype_to_jax(self.config.param_dtype)
        self.scale = self.param(
            "scale", self.config.scale_init.instantiate(InitInterface), (self.config.input_dim,), param_dtype
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Cast to computation dtype
        dtype = str_dtype_to_jax(self.config.dtype)
        x = x.astype(dtype)
        scale = self.scale.astype(dtype)

        return (scale[None, :] * x.reshape(-1, self.config.input_dim)).reshape(x.shape)
