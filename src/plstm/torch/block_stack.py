import torch
import torch.nn as nn

from ..config.block_stack import BlockStackConfig
from .interfaces import ResidualModule
from .blocks import *  # noqa
from .vision_blocks import *  # noqa
from .transformer_block import *  # noqa
from compoconf import register


@register
class BlockStack(ResidualModule):
    """Inner module that handles block creation and processing."""

    def __init__(self, config: BlockStackConfig):
        ResidualModule.__init__(self, config)
        self.config = config

        # Create blocks using ModuleList for proper parameter registration
        self.blocks = nn.ModuleList(
            [self.config.block.instantiate(ResidualModule) for _ in range(self.config.num_blocks)]
        )

    def reset_parameters(self):
        for block in self.blocks:
            block.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the blocks."""
        for block in self.blocks:
            x = block(x, **kwargs)
        return x
