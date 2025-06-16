import torch
from compoconf import RegistrableConfigInterface, register_interface
from ..config.interfaces import ResidualModuleConfig, ModuleConfig
from abc import abstractmethod


@register_interface
class ResidualModule(torch.nn.Module, RegistrableConfigInterface):
    config: ResidualModuleConfig

    def __init__(self, config: ResidualModuleConfig, **kwargs):
        torch.nn.Module.__init__(self)
        RegistrableConfigInterface.__init__(self, config)

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class Module(torch.nn.Module):
    config: ModuleConfig

    def __init__(self, config: ModuleConfig, **kwargs):
        torch.nn.Module.__init__(self)
        RegistrableConfigInterface.__init__(self, config)

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
