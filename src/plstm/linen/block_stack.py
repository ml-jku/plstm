import jax
from flax import linen as nn

from ..config.block_stack import BlockStackConfig
from .interfaces import ResidualModule
from .blocks import *  # noqa
from .vision_blocks import *  # noqa
from .transformer_block import *  # noqa
from compoconf import register
from .dtype import str_dtype_to_jax


@register
class BlockStack(ResidualModule):
    """Inner module that handles block creation and processing."""

    config: BlockStackConfig

    def setup(self):
        if self.config.scan_blocks:
            self.block = self.config.block.instantiate(ResidualModule)

        else:
            # Create blocks explicitly for module inspection
            self.blocks = [self.config.block.instantiate(ResidualModule) for _ in range(self.config.num_blocks)]

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        """Forward pass through the blocks.

        Supports both explicit block creation and scan-based processing.
        """
        x = x.astype(str_dtype_to_jax(self.config.dtype))
        if self.config.scan_blocks:
            # Use scan to process the blocks
            def scan_fn(block, x, _):
                return block(x, **kwargs), None

            # Create a scan module
            y, _ = nn.scan(
                scan_fn,
                split_rngs={"params": True, "dropout": True},
                variable_axes={"params": 0, "intermediates": 0},
                length=self.config.num_blocks,
            )(self.block, x, None)

            # We need to create a dummy carry for scan
            # Run the scan
            return y
        else:
            # Process blocks sequentially
            for block in self.blocks:
                x = block(x, **kwargs)
            return x
