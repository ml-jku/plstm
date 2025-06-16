from dataclasses import dataclass
from ..util import assert_check_literals
from .dtype import DTYPES


@dataclass
class Convolution1DLayerConfig:
    input_dim: int = -1
    output_dim: int = -1
    causal: bool = True
    kernel_size: int = 4
    pointwise: bool = True  # currently only option
    bias: bool = True

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert self.input_dim > 0
        if self.output_dim < 0:
            self.output_dim = self.input_dim
        assert self.kernel_size > 0

        assert_check_literals(self)


@dataclass
class Convolution2DLayerConfig:
    input_dim: int = -1
    output_dim: int = -1
    causal: bool = False
    kernel_size: tuple[int, int] = (3, 3)
    pointwise: bool = True  # currently only option
    bias: bool = True

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert self.input_dim > 0
        if self.output_dim < 0:
            self.output_dim = self.input_dim
        assert len(self.kernel_size) == 2
        assert self.kernel_size[0] > 0
        assert self.kernel_size[1] > 0

        assert_check_literals(self)
