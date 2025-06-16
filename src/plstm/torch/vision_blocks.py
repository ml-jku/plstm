from compoconf import register
from ..config.vision_blocks import pLSTMVisionBlockConfig1
from .interfaces import ResidualModule
from torch import nn


@register
class pLSTMVisionBlock1(ResidualModule):
    config: pLSTMVisionBlockConfig1

    def __init__(self, config: pLSTMVisionBlockConfig1):
        ResidualModule.__init__(self, config)
        self.config = config
        self.block0 = config.block0.instantiate(ResidualModule)
        self.block1 = config.block1.instantiate(ResidualModule)
        self.drop_path = nn.Dropout(config.drop_path_rate)

    def reset_parameters(self, *args, **kwargs):
        self.block0.reset_parameters()
        self.block1.reset_parameters()

    def forward(self, x, deterministic: bool = False):
        x1 = self.block0(x, deterministic=deterministic)
        x2 = self.block1(x1, deterministic=deterministic)
        if deterministic:
            return x2
        else:
            return self.drop_path(x.new_ones((x.shape[0],))).view((-1, *[1] * (x.ndim - 1))) * (x2 - x) + x
