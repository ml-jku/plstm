from dataclasses import dataclass, field
from typing import Literal
from .blocks import PreUpProjectionBlockConfig, PostUpProjectionBlockConfig
from .scalar import SoftCapFunctionConfig
from .norm import RMSNormConfig
from .block_stack import BlockStackConfig
from compoconf import ConfigInterface
from .dtype import DTYPES
from ..util import assert_check_literals
from .initialization import WeightInitConfig, SmallInitConfig


@dataclass
class pLSTMLMModelConfig(ConfigInterface):
    _shortname: str = "pLSTMLM"
    vocab_size: int | None = -1
    input_dim: int | None = -1
    output_dim: int | None = -1
    context_length: int | None = -1
    embedding_dim: int | None = -1
    num_blocks: int = 1
    tie_weights: bool = False
    logit_soft_cap: SoftCapFunctionConfig | None = None

    block_type: Literal["pre_up", "post_up"] = "pre_up"
    block_stack: BlockStackConfig | None = None

    embed_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    head_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    post_blocks_norm: RMSNormConfig | None = None

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert self.vocab_size > 0
        assert self.embedding_dim > 0
        self.input_dim = self.embedding_dim
        self.output_dim = self.input_dim
        if self.post_blocks_norm is None:
            self.post_blocks_norm = RMSNormConfig(
                input_dim=self.input_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        if self.logit_soft_cap is None:
            self.logit_soft_cap = SoftCapFunctionConfig(
                scale=10.0,
            )
        if self.block_stack is None:
            if self.block_type == "pre_up":
                self.block_stack = BlockStackConfig(
                    block=PreUpProjectionBlockConfig(
                        input_dim=self.input_dim,
                        interaction_module_name="pLSTM1D",
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    num_blocks=self.num_blocks,
                    input_dim=self.input_dim,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            elif self.block_type == "post_up":
                self.block_stack = BlockStackConfig(
                    block=PostUpProjectionBlockConfig(
                        input_dim=self.input_dim,
                        interaction_module_name="pLSTM1D",
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    num_blocks=self.num_blocks,
                    input_dim=self.input_dim,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
        else:
            assert (self.block_type == "pre_up" and isinstance(self.block_stack.block, PreUpProjectionBlockConfig)) or (
                self.block_type == "post_up" and isinstance(self.block_stack.block, PostUpProjectionBlockConfig)
            )
        assert_check_literals(self)
