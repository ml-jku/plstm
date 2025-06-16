from compoconf import register_interface, register, RegistrableConfigInterface
from plstm.nnx_dummy import nnx
from ..config.scalar import (
    ScalarFunctionConfig,
    SigmoidFunctionConfig,
    TanhFunctionConfig,
    SoftCapFunctionConfig,
    ExpExpFunctionConfig,
)
from abc import abstractmethod
import jax.numpy as jnp
import jax.lax


@register_interface
class ScalarFunctionLayer(nnx.Module, RegistrableConfigInterface):
    config: ScalarFunctionConfig

    def __init__(self, config: ScalarFunctionConfig, **kwargs):
        nnx.Module.__init__(self)
        RegistrableConfigInterface.__init__(self, config)
        self.config = config

    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        """Input and output shape of this function are the same.

        It should act componentwise only.
        """
        raise NotImplementedError


@register
class TanhFunctionLayer(ScalarFunctionLayer):
    config: TanhFunctionConfig

    def __init__(self, config, **kwargs):
        ScalarFunctionLayer.__init__(self, config)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.tanh(x)


@register
class SigmoidFunctionLayer(ScalarFunctionLayer):
    config: SigmoidFunctionConfig

    def __init__(self, config, **kwargs):
        ScalarFunctionLayer.__init__(self, config)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.sigmoid(x)


@register
class ExpExpFunctionLayer(ScalarFunctionLayer):
    config: ExpExpFunctionConfig

    def __init__(self, config, **kwargs):
        ScalarFunctionLayer.__init__(self, config)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.exp(-self.config.scale * jnp.exp(x))


@register
class SoftCapFunctionLayer(ScalarFunctionLayer):
    config: SoftCapFunctionConfig

    def __init__(self, config, **kwargs):
        ScalarFunctionLayer.__init__(self, config)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.config.scale * jnp.tanh(x / self.config.scale)
