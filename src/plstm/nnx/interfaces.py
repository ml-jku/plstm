from plstm.nnx_dummy import nnx
from compoconf import RegistrableConfigInterface, register_interface
from ..config.interfaces import ResidualModuleConfig, ModuleConfig
from abc import abstractmethod
import jax


@register_interface
class Module(nnx.Module, RegistrableConfigInterface):
    config: ModuleConfig

    def __init__(self, config, **kwargs):
        nnx.Module.__init__(self)
        RegistrableConfigInterface.__init__(self, config)

    @abstractmethod
    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        raise NotImplementedError


@register_interface
class ResidualModule(Module):
    config: ResidualModuleConfig

    def __init__(self, config, **kwargs):
        nnx.Module.__init__(self)
        RegistrableConfigInterface.__init__(self, config)

    @abstractmethod
    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        raise NotImplementedError
