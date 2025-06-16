from compoconf import register_interface, register, RegistrableConfigInterface
import torch.nn as nn
from ..config.scalar import (
    ScalarFunctionConfig,
    SigmoidFunctionConfig,
    TanhFunctionConfig,
    SoftCapFunctionConfig,
    ExpExpFunctionConfig,
)
from abc import abstractmethod
import torch


@register_interface
class ScalarFunctionLayer(nn.Module, RegistrableConfigInterface):
    config: ScalarFunctionConfig

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input and output shape of this function are the same.

        It should act componentwise only.
        """
        raise NotImplementedError


@register
class TanhFunctionLayer(ScalarFunctionLayer):
    config: TanhFunctionConfig

    def __init__(self, config, **kwargs):
        nn.Module.__init__(self)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


@register
class SigmoidFunctionLayer(ScalarFunctionLayer):
    config: SigmoidFunctionConfig

    def __init__(self, config, **kwargs):
        nn.Module.__init__(self)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


@register
class ExpExpFunctionLayer(ScalarFunctionLayer):
    config: ExpExpFunctionConfig

    def __init__(self, config, **kwargs):
        nn.Module.__init__(self)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.config.scale * torch.exp(x))


@register
class SoftCapFunctionLayer(ScalarFunctionLayer):
    config: SoftCapFunctionConfig

    def __init__(self, config, **kwargs):
        nn.Module.__init__(self)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.config.scale * torch.tanh(x / self.config.scale)
