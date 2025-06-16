from dataclasses import dataclass, field
from .dtype import DTYPES
from .initialization import ScaleInitConfig, OnesInitConfig
from compoconf import ConfigInterface


@dataclass
class ScaleLayerConfig(ConfigInterface):
    input_dim: int = -1
    output_dim: int = -1
    scale: float = 1.0  # deprecated and not used anymore!
    scale_init: ScaleInitConfig = field(default_factory=OnesInitConfig)

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert self.input_dim > 0
        if self.output_dim < 0:
            self.output_dim = self.input_dim
        assert self.input_dim == self.output_dim
