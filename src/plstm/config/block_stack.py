from dataclasses import dataclass
from ..util import assert_check_literals
from .interfaces import ResidualModuleConfig
from .blocks import PreUpProjectionBlockConfig, PostUpProjectionBlockConfig
from .vision_blocks import pLSTMVisionBlockConfig1
from .transformer_block import TransformerBlockConfig


@dataclass
class BlockStackConfig(ResidualModuleConfig):
    input_dim: int = -1
    output_dim: int = -1

    block: (
        PreUpProjectionBlockConfig
        | PostUpProjectionBlockConfig
        | pLSTMVisionBlockConfig1
        | TransformerBlockConfig
        | None
    ) = None

    scan_blocks: bool = False
    num_blocks: int = -1

    def __post_init__(self):
        assert self.block is not None
        assert self.input_dim > 0
        if self.output_dim < 0:
            self.output_dim = self.input_dim
        assert self.output_dim == self.input_dim
        if self.block.input_dim < 0:
            self.block.input_dim = self.input_dim

        self.block.__post_init__()
        assert_check_literals(self)
