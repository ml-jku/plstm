from compoconf import register
from ..config.graph_block import pLSTMGraphBlockConfig, pLSTMGraphEdgeBlockConfig
from .interfaces import ResidualModule


@register
class pLSTMGraphBlock(ResidualModule):
    config: pLSTMGraphBlockConfig

    def __init__(self, config: pLSTMGraphBlockConfig):
        ResidualModule.__init__(self, config)
        self.config = config
        self.block0 = config.block0.instantiate(ResidualModule)
        self.block1 = config.block1.instantiate(ResidualModule)

    def reset_parameters(self, *args, **kwargs):
        self.block0.reset_parameters()
        self.block1.reset_parameters()

    def forward(self, x, deterministic: bool = False, **kwargs):
        x = self.block0(x, deterministic=deterministic, **kwargs)
        x = self.block1(x, deterministic=deterministic, **kwargs)
        return x


@register
class pLSTMGraphEdgeBlock(ResidualModule):
    config: pLSTMGraphEdgeBlockConfig

    def __init__(self, config: pLSTMGraphBlockConfig):
        ResidualModule.__init__(self, config)
        self.config = config
        self.block0 = config.block0.instantiate(ResidualModule)
        self.block1 = config.block1.instantiate(ResidualModule)

    def reset_parameters(self, *args, **kwargs):
        self.block0.reset_parameters()
        self.block1.reset_parameters()

    def forward(self, x, deterministic: bool = False, **kwargs):
        x = self.block0(x, deterministic=deterministic, **kwargs)
        x = self.block1(x, deterministic=deterministic, **kwargs)
        return x
