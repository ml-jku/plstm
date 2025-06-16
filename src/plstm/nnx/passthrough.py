import jax
from plstm.nnx_dummy import nnx

from ..config.passthrough import PassthroughLayerConfig
from .dtype import str_dtype_to_jax
from .initialization import InitInterface


class PassthroughLayer(nnx.Module):
    """Layer that passes through input with learned scale."""

    config: PassthroughLayerConfig

    def __init__(self, config: PassthroughLayerConfig, *, rngs: nnx.Rngs):
        nnx.Module.__init__(self)
        self.config = config
        param_dtype = str_dtype_to_jax(config.param_dtype)
        self.scale = nnx.Param(
            config.scale_init.instantiate(InitInterface)(rngs.params(), (self.config.input_dim,), dtype=param_dtype)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Cast to computation dtype
        dtype = str_dtype_to_jax(self.config.dtype)
        x = x.astype(dtype)
        scale = self.scale.astype(dtype)

        return (scale[None, :] * x.reshape(-1, self.config.input_dim)).reshape(x.shape)
