from dataclasses import dataclass, field
from typing import Literal
from ..util import assert_check_literals
from .dtype import DTYPES
from .initialization import BiasInitConfig, ConstantInitConfig, WeightInitConfig, ZerosInitConfig


@dataclass
class SourceLayerConfig:
    num_heads: int = 1
    sub_heads: int = 1
    JK: int = 1
    JV: int = 1
    JT: int = 1
    input_dim: int = -1

    bias_init: BiasInitConfig = field(default_factory=lambda: ConstantInitConfig(-5.0))
    weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)
    weight: bool = False
    activation: Literal["identity", "tanh", "logsigmoid"] = "logsigmoid"
    activation_scale: float = 1.0

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert self.input_dim > 0, "Input Dim must be set"
        assert self.num_heads > 0
        assert self.sub_heads > 0
        assert self.JK > 0
        assert self.JV > 0
        assert self.JT > 0

        assert_check_literals(self)


@dataclass
class MarkLayerConfig:
    num_heads: int = 1
    JQ: int = 1
    JO: int = 1
    JT: int = 1
    input_dim: int = -1
    sub_heads: int = 1

    bias_init: BiasInitConfig = field(default_factory=lambda: ConstantInitConfig(-5.0))
    weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)
    weight: bool = False
    activation: Literal["identity", "tanh", "logsigmoid"] = "logsigmoid"
    activation_scale: float = 1.0

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert self.input_dim > 0, "Input Dim must be set"
        assert self.num_heads > 0
        assert self.JQ > 0
        assert self.JO > 0
        assert self.JT > 0
        assert self.sub_heads > 0

        assert_check_literals(self)


@dataclass
class DirectLayerConfig:
    num_heads: int = 1
    JQ: int = 1
    JK: int = 1
    JO: int = 1
    JV: int = 1
    input_dim: int = -1
    sub_heads: int = 1

    bias_init: BiasInitConfig = field(default_factory=lambda: ConstantInitConfig(-5.0))
    weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)
    weight: bool = False
    activation: Literal["identity", "tanh", "logsigmoid"] = "logsigmoid"
    activation_scale: float = 1.0

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert self.input_dim > 0, "Input Dim must be set"
        assert self.num_heads > 0
        assert self.JQ > 0
        assert self.JK > 0
        assert self.JO > 0
        assert self.JV > 0
        assert self.sub_heads > 0

        assert_check_literals(self)
