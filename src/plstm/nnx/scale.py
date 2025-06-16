from plstm.nnx_dummy import nnx
import jax.numpy as jnp

from ..config.scale import ScaleLayerConfig
from .interfaces import ResidualModule
from .dtype import str_dtype_to_jax
from .initialization import InitInterface
from compoconf import register


@register
class ScaleLayer(ResidualModule):
    config: ScaleLayerConfig

    def __init__(self, config: ScaleLayerConfig, rngs: nnx.Rngs, **kwargs):
        ResidualModule.__init__(self, config)
        self.config = config
        param_dtype = str_dtype_to_jax(config.param_dtype)
        scale_shape = [config.input_dim]
        self.scale = nnx.Param(
            self.config.scale_init.instantiate(InitInterface)(rngs.params(), scale_shape, dtype=param_dtype)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Cast to computation dtype
        dtype = str_dtype_to_jax(self.config.dtype)
        x = x.astype(dtype)
        scale = self.scale.astype(dtype)

        # Create tuple of 1s for reshape
        ones_tuple = (1,) * (x.ndim - 1)
        return scale.reshape(*ones_tuple, *scale.shape) * x
