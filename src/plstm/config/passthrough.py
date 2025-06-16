from dataclasses import dataclass, field
from ..util import assert_check_literals
from .dtype import DTYPES
from .initialization import ScaleInitConfig, OnesInitConfig


@dataclass
class PassthroughLayerConfig:
    input_dim: int = -1
    output_dim: int = -1

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"
    scale_init: ScaleInitConfig = field(default_factory=OnesInitConfig)

    def __post_init__(self):
        assert self.input_dim > 0
        if self.output_dim < 0:
            self.output_dim = self.input_dim
        assert self.output_dim == self.input_dim

        assert_check_literals(self)
