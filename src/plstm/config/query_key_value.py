from dataclasses import dataclass, field
from typing import Literal
from .dtype import DTYPES
from ..util import assert_check_literals
from .initialization import BiasInitConfig, WeightInitConfig, ZerosInitConfig, SmallInitConfig


@dataclass
class QueryLayerConfig:
    num_heads: int = 1  # output heads
    sub_heads: int = 1  # can be used to make weight matrix smaller
    input_dim: int = -1
    DK: int = -1

    JQ: int = 1
    bias: bool = True
    normalization: Literal["none", "l2"] = "none"
    bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)
    weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert_check_literals(self)
        assert self.num_heads > 0
        assert self.input_dim > 0
        # assert self.sub_heads > 0
        assert self.DK > 0

        if self.sub_heads < 0:
            self.sub_heads = self.num_heads

        # For eye init, we need:
        # 1. input_dim == JQ * DK (original condition)
        # 2. sub_heads == num_heads (to ensure weight matrix is square)
        if self.weight_init == "eye":
            assert self.input_dim == self.num_heads * self.JQ * self.DK, "For eye init, input_dim must equal JQ * DK"


@dataclass
class KeyLayerConfig:
    num_heads: int = 1  # output_heads
    sub_heads: int = 1  # can be used to make weight matrix smaller
    input_dim: int = -1
    DK: int = -1

    JK: int = 1
    bias: bool = True
    normalization: Literal["none", "l2"] = "none"
    bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)
    weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert_check_literals(self)
        assert self.num_heads > 0
        assert self.input_dim > 0
        assert self.sub_heads > 0
        assert self.DK > 0

        if self.sub_heads < 0:
            self.sub_heads = self.num_heads

        # For eye init, we need:
        # 1. input_dim == JK * DK (original condition)
        # 2. sub_heads == num_heads (to ensure weight matrix is square)
        if self.weight_init == "eye":
            assert self.input_dim == self.num_heads * self.JK * self.DK, "For eye init, input_dim must equal JK * DK"


@dataclass
class ValueLayerConfig:
    num_heads: int = 1  # output_heads
    sub_heads: int = 1  # can be used to make weight matrix smaller
    input_dim: int = -1
    DV: int = -1

    JV: int = 1
    bias: bool = True
    normalization: Literal["none", "l2"] = "none"
    bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)
    weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert_check_literals(self)
        assert self.num_heads > 0
        assert self.input_dim > 0
        if self.sub_heads < 0:
            self.sub_heads = self.num_heads
        assert self.DV > 0

        # For eye init, we need:
        # 1. input_dim == JV * DV (original condition)
        # 2. sub_heads == num_heads (to ensure weight matrix is square)
        if self.weight_init == "eye":
            assert self.input_dim == self.num_heads * self.JV * self.DV, "For eye init, input_dim must equal JV * DV"
