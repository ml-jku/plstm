import torch
from torch import nn

from ..config.passthrough import PassthroughLayerConfig
from .initialization import InitInterface


class PassthroughLayer(nn.Module):
    config: PassthroughLayerConfig

    def __init__(self, config: PassthroughLayerConfig):
        nn.Module.__init__(self)
        self.config = config
        self.scale = nn.Parameter(torch.zeros(self.config.input_dim))
        self.reset_parameters()

    def reset_parameters(self, *args, **kwargs):
        with torch.no_grad():
            self.config.scale_init.instantiate(InitInterface)(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.scale[None, :] * x.reshape(-1, self.config.input_dim)).view(x.shape)
