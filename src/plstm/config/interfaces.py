from dataclasses import dataclass
from compoconf import ConfigInterface

from ..util import assert_check_literals
from .dtype import DTYPES


@dataclass
class ModuleConfig(ConfigInterface):
    input_dim: int = -1
    output_dim: int = -1

    param_dtype: DTYPES = "float32"
    dtype: DTYPES = "bfloat16"

    def __post_init__(self):
        assert self.input_dim > 0
        assert self.output_dim > 0
        assert_check_literals(self)


@dataclass
class ResidualModuleConfig(ModuleConfig):
    input_dim: int = -1
    output_dim: int = -1

    param_dtype: DTYPES = "float32"
    dtype: DTYPES = "bfloat16"

    def __post_init__(self):
        assert self.input_dim > 0
        if self.output_dim < 0:
            self.output_dim = self.input_dim
        assert_check_literals(self)
