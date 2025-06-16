import jax
from plstm.nnx_dummy import nnx

from ..config.block_stack import BlockStackConfig
from .interfaces import ResidualModule
from .blocks import *  # noqa
from .vision_blocks import *  # noqa
from .transformer_block import *  # noqa
from compoconf import register
# from flax.nnx import reprlib


@register
class BlockStack(ResidualModule):
    """Inner module that handles block creation and processing."""

    config: BlockStackConfig

    def __init__(self, config: BlockStackConfig, rngs: nnx.Rngs):
        ResidualModule.__init__(self, config)
        self.config = config
        # if self.config.scan_blocks:
        #     raise NotImplementedError
        # else:
        # Create blocks explicitly for module inspection
        for i in range(self.config.num_blocks):
            block = self.config.block.instantiate(ResidualModule, rngs=rngs)
            setattr(self, f"block{i}", block)

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        """Forward pass through the blocks.

        Supports both explicit block creation and scan-based processing.
        """
        # if self.config.scan_blocks:
        #     raise NotImplementedError
        # else:
        for i in range(self.config.num_blocks):
            x = getattr(self, f"block{i}")(x, **kwargs)
        return x

    # To be optimized
    # def __nnx_repr__(self):
    #     yield reprlib.Object(type=type(self))
