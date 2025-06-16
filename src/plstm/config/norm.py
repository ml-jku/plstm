from dataclasses import dataclass, field
from ..util import assert_check_literals
from .dtype import DTYPES
from compoconf import ConfigInterface
from .initialization import BiasInitConfig, ScaleInitConfig, ZerosInitConfig, OnesInitConfig


@dataclass
class MultiHeadNormLayerConfig:
    num_heads: int = -1
    input_dim: int = -1
    output_dim: int = -1

    eps: float = 1e-5
    dtype: DTYPES = "float32"
    param_dtype: DTYPES = "float32"
    # axis of the heads to be vmapped over
    axis: int = 1
    bias: bool = False
    scale: bool = True
    bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)
    scale_init: ScaleInitConfig = field(default_factory=OnesInitConfig)

    def __post_init__(self):
        if self.output_dim < 0:
            self.output_dim = self.input_dim
        assert self.input_dim > 0
        assert self.num_heads > 0

        assert_check_literals(self)


@dataclass
class MultiHeadRMSNormConfig(MultiHeadNormLayerConfig, ConfigInterface):
    pass


@dataclass
class MultiHeadLayerNormConfig(MultiHeadNormLayerConfig, ConfigInterface):
    pass


@dataclass
class NormLayerConfig:
    input_dim: int = -1
    output_dim: int = -1

    eps: float = 1e-5
    dtype: DTYPES = "float32"
    param_dtype: DTYPES = "float32"
    bias: bool = False
    scale: bool = True
    bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)
    scale_init: ScaleInitConfig = field(default_factory=OnesInitConfig)

    def __post_init__(self):
        if self.output_dim < 0:
            self.output_dim = self.input_dim
        assert self.input_dim > 0

        assert_check_literals(self)


@dataclass
class LayerNormConfig(NormLayerConfig, ConfigInterface):
    pass


@dataclass
class RMSNormConfig(NormLayerConfig, ConfigInterface):
    pass


@dataclass
class IdentityConfig(ConfigInterface):
    pass


NormConfig = LayerNormConfig | RMSNormConfig | IdentityConfig | MultiHeadRMSNormConfig | MultiHeadLayerNormConfig
