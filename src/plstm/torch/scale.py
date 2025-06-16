import torch
from torch import nn

from ..config.scale import ScaleLayerConfig
from .initialization import InitInterface
from .interfaces import ResidualModule
from compoconf import register


@register
class ScaleLayer(ResidualModule):
    config: ScaleLayerConfig

    def __init__(self, config: ScaleLayerConfig):
        nn.Module.__init__(self)
        self.config = config
        self.scale = nn.Parameter(torch.ones(self.config.input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.config.scale_init.instantiate(InitInterface)(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create tuple of 1s for reshape
        ones_tuple = (1,) * (x.ndim - 1)
        return self.scale.reshape(*ones_tuple, *self.scale.shape) * x
