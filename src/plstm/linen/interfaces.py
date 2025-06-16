from flax import linen as nn
from compoconf import RegistrableConfigInterface, register_interface
from ..config.interfaces import ResidualModuleConfig, ModuleConfig
from abc import abstractmethod
import jax


@register_interface
class Module(nn.Module, RegistrableConfigInterface):
    config: ModuleConfig

    @abstractmethod
    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        raise NotImplementedError


@register_interface
class ResidualModule(Module):
    config: ResidualModuleConfig

    @abstractmethod
    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        raise NotImplementedError
