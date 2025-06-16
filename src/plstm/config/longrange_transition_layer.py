from dataclasses import dataclass, field
from typing import Literal
from ..util import assert_check_literals
from .dtype import DTYPES
from .initialization import BiasInitConfig, WeightInitConfig, ZerosInitConfig, DiagonalInitConfig, OnesInitConfig


@dataclass
class LongRangeTransitionLayerConfig:
    # output shape is (..., num_heads, transition_dim, transition_dim)
    num_heads: int = 1
    transition_dim: int = -1
    input_dim: int = -1
    sub_heads: int = 1

    inproj_bias_init: BiasInitConfig = field(default_factory=lambda: DiagonalInitConfig(in_axes=-2, out_axes=-1))
    outproj_bias_init: BiasInitConfig = field(default_factory=lambda: DiagonalInitConfig(in_axes=-2, out_axes=-1))
    inproj_weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)
    outproj_weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)

    eigenvalue_bias_init: BiasInitConfig = field(default_factory=OnesInitConfig)
    eigenvalue_weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)
    eigenvalue_representation: Literal["logsigmoid", "expexp", "tanh"] = "tanh"
    eigenvalue_factor: float = 5
    symmetric: bool = False
    orthogonalization_factor: float = 1.0
    orthogonalization_order: int = 16

    weight: bool = True

    normalization_mode: Literal[
        "qr",
        "eigenvalue_restriction",
        "singularvalue_restriction",
        "householder_orthogonalization",
        "exponential_orthogonalization",
    ] = "exponential_orthogonalization"

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert self.transition_dim > 0
        assert self.input_dim > 0
        assert self.num_heads > 0
        assert self.sub_heads > 0

        assert_check_literals(self)
