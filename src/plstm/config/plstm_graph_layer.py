from typing import Literal
from dataclasses import dataclass, field
from .interfaces import ResidualModuleConfig
from .dtype import DTYPES
from .initialization import (
    BiasInitConfig,
    ZerosInitConfig,
    ConstantInitConfig,
    WeightInitConfig,
    SmallInitConfig,
    DiagonalInitConfig,
    NormalInitConfig,
)
from .norm import MultiHeadRMSNormConfig


@dataclass
class pLSTMGraphLayerConfig(ResidualModuleConfig):
    # for the P mode there are actually two options: L1 and Linf normalized
    # transitions, so row or column stochastic (for the absolute values)
    mode: Literal["D", "P"] = "P"
    memeff_variant: bool = True

    input_dim: int = -1
    output_dim: int = -1
    num_heads: int = -1
    qk_head_dim: int = -1
    hv_head_dim: int = -1

    max_edges: int = -1

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    bias: bool = False
    qkvo_bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)
    qkvo_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)

    source_bias_init: BiasInitConfig = field(default_factory=lambda: ConstantInitConfig(value=-5.0))
    source_weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)
    transition_bias_init: BiasInitConfig = field(
        default_factory=lambda: DiagonalInitConfig(value=10.0, in_axes=-1, out_axes=-2)
    )
    transition_weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)
    mark_bias_init: BiasInitConfig = field(default_factory=lambda: ConstantInitConfig(value=-5.0))
    mark_weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)
    direct_bias_init: BiasInitConfig = field(default_factory=lambda: ConstantInitConfig(value=-5.0))
    direct_weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)

    mhnorm: MultiHeadRMSNormConfig | None = None
    out: bool = True

    def __post_init__(self):
        assert self.input_dim > 0
        assert self.num_heads > 0
        assert self.max_edges > 0

        if self.output_dim < 0:
            self.output_dim = self.input_dim

        if self.qk_head_dim < 0:
            self.qk_head_dim = self.input_dim // self.num_heads
        if self.hv_head_dim < 0:
            self.hv_head_dim = self.input_dim // self.num_heads
        if self.mhnorm is None:
            self.mhnorm = MultiHeadRMSNormConfig(input_dim=self.input_dim, num_heads=self.num_heads, axis=0)


@dataclass
class pLSTMGraphEdgeLayerConfig(pLSTMGraphLayerConfig):
    edge_input_dim: int = -1
    transition_bias_init: BiasInitConfig = field(default_factory=lambda: NormalInitConfig(stddev=10.0))
    transition_weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)
    max_edges: int = 100000000  # not used

    def __post_init__(self):
        super().__post_init__()
        assert self.edge_input_dim > 0
