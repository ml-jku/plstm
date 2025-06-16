from dataclasses import dataclass, field
from typing import Literal
from .dtype import DTYPES

from .norm import NormConfig, RMSNormConfig
from ..util import assert_check_literals
from .interfaces import ResidualModuleConfig
from .scale import ScaleLayerConfig
from .initialization import BiasInitConfig, WeightInitConfig, SmallInitConfig, ZerosInitConfig


@dataclass
class TransformerBlockConfig(ResidualModuleConfig):
    class_name: str = "TransformerBlock"
    input_dim: int = -1
    output_dim: int = -1
    num_heads: int = 1

    projection_scaling: float = 4  # 8 / 3
    projection_round: int = 64
    gated: bool = False
    bias: bool = True

    upproj_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    downproj_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    upproj_bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)
    downproj_bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)
    attn_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    out_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)

    # potentially used as activation function in MLP if not gated
    gating_function: Literal["silu", "gelu"] = "gelu"
    norm: NormConfig = None
    scale: ScaleLayerConfig | None = None
    drop_path_rate: float = 0.0
    skip: bool = True
    decode: bool = False

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    @property
    def inner_input_dim(self):
        return (
            int(round((self.projection_scaling * self.input_dim - 1) // self.projection_round) + 1)
            * self.projection_round
        )

    def __post_init__(self):
        assert_check_literals(self)
        assert self.input_dim > 0
        assert self.num_heads > 0
        assert 1.0 >= self.drop_path_rate >= 0.0
        if self.norm is None:
            self.norm = RMSNormConfig(input_dim=self.input_dim)
        if self.output_dim <= 0:
            self.output_dim = self.input_dim
        if self.scale is None:
            self.scale = ScaleLayerConfig(input_dim=self.input_dim)
