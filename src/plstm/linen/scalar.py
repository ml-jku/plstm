from compoconf import register_interface, register, RegistrableConfigInterface
from flax import linen as nn
from ..config.scalar import (
    ScalarFunctionConfig,
    SigmoidFunctionConfig,
    TanhFunctionConfig,
    SoftCapFunctionConfig,
    ExpExpFunctionConfig,
)
from abc import abstractmethod
import jax.numpy as jnp
import jax


@register_interface
class ScalarFunctionLayer(nn.Module, RegistrableConfigInterface):
    config: ScalarFunctionConfig

    def __post_init__(self):
        super().__post_init__()
        RegistrableConfigInterface.__init__(self, self.config)

    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        """Input and output shape of this function are the same.

        It should act componentwise only.
        """
        raise NotImplementedError


@register
class TanhFunctionLayer(ScalarFunctionLayer):
    config: TanhFunctionConfig

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.tanh(x)


@register
class SigmoidFunctionLayer(ScalarFunctionLayer):
    config: SigmoidFunctionConfig

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.sigmoid(x)


@register
class ExpExpFunctionLayer(ScalarFunctionLayer):
    config: ExpExpFunctionConfig

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.exp(-self.config.scale * jnp.exp(x))


@register
class SoftCapFunctionLayer(ScalarFunctionLayer):
    config: SoftCapFunctionConfig

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.config.scale * jnp.tanh(x / self.config.scale)
