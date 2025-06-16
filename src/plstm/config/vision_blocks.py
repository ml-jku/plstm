from dataclasses import dataclass, field
from typing import Literal
from .interfaces import ResidualModuleConfig
from .blocks import PreUpProjectionBlockConfig, PostUpProjectionBlockConfig
from .plstm_2d_layer_fused import pLSTM2DLayerFusedConfig
from ..util import assert_check_literals
from .dtype import DTYPES


@dataclass
class pLSTMVisionBlockConfig(ResidualModuleConfig):
    pass


@dataclass
class pLSTMVisionBlockConfig1(pLSTMVisionBlockConfig):
    input_dim: int = 32
    num_heads: int = 8
    block_type: Literal["pre_up", "post_up"] = "post_up"
    block_mode: Literal["PD", "DP"] = "PD"
    orientations0: list[int] = field(default_factory=lambda: [0, 1, 2, 3])  # deprecated
    orientations1: list[int] = field(default_factory=lambda: [0, 1, 2, 3])  # deprecated
    # ideally the type annotation should be sth like Block.interfaces, maybe it is necessary to reflect
    # the common structure of interfaces with dummy modules without backend
    block0: PreUpProjectionBlockConfig | PostUpProjectionBlockConfig | None = None
    block1: PreUpProjectionBlockConfig | PostUpProjectionBlockConfig | None = None

    drop_path_rate: float = 0.0

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert_check_literals(self)
        assert self.input_dim > 0
        assert self.num_heads > 0
        assert 1.0 >= self.drop_path_rate >= 0.0
        if self.block0 is None:
            self.block0 = PostUpProjectionBlockConfig(
                input_dim=self.input_dim,
                interaction_module_name="pLSTM2DFused",
                interaction_module=pLSTM2DLayerFusedConfig(
                    mode=self.block_mode[0],
                    input_dim=self.input_dim,
                    num_heads=self.num_heads,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                gated=False,
            )

        if self.block1 is None:
            self.block1 = PostUpProjectionBlockConfig(
                input_dim=self.input_dim,
                interaction_module_name="pLSTM2DFused",
                interaction_module=pLSTM2DLayerFusedConfig(
                    mode=self.block_mode[1],
                    input_dim=self.input_dim,
                    num_heads=self.num_heads,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                gated=False,
            )
