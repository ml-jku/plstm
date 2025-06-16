from dataclasses import dataclass
from compoconf import ConfigInterface

# these initialization configurations should cover both weights/kernels and biases/scales


@dataclass
class ConstantInitConfig(ConfigInterface):
    value: float = 0.0


@dataclass
class OnesInitConfig(ConfigInterface):
    pass


@dataclass
class ZerosInitConfig(ConfigInterface):
    pass


@dataclass
class LinspaceInitConfig(ConfigInterface):
    low: float = 0.0
    high: float = 1.0
    axis: int = -1


@dataclass
class LogspaceInitConfig(ConfigInterface):
    low: float = 0.0
    high: float = 1.0
    base: float = 10.0
    axis: int = -1


@dataclass
class NormalInitConfig(ConfigInterface):
    mean: float = 0.0
    stddev: float = 1.0


@dataclass
class TruncatedNormalInitConfig(ConfigInterface):
    mean: float = 0.0
    stddev: float = 0.0
    lower: float = -2.0
    upper: float = 2.0


@dataclass
class DiagonalInitConfig(ConfigInterface):
    value: float = 1.0
    in_axes: int | tuple[int, ...] = ()
    out_axes: int | tuple[int, ...] = ()

    def __post_init__(self):
        ia, oa = (((ax,) if isinstance(ax, int) else ax) for ax in (self.in_axes, self.out_axes))
        assert not set(ia).intersection(set(oa)), "Axes should be disjoint"


# see
# https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py


@dataclass
class WangInitConfig(ConfigInterface):
    num_blocks: int = 1
    mup_init_scale: float = 1.0
    axis: int = -1


@dataclass
class XavierInitConfig(ConfigInterface):
    in_axis: int = -2
    out_axis: int = -1
    mup_init_scale: float = 1.0


@dataclass
class SmallInitConfig(ConfigInterface):
    axis: int = -1
    mup_init_scale: float = 1.0


BiasInitConfig = (
    ConstantInitConfig
    | ZerosInitConfig
    | OnesInitConfig
    | LinspaceInitConfig
    | NormalInitConfig
    | TruncatedNormalInitConfig
    | DiagonalInitConfig
)

WeightInitConfig = (
    WangInitConfig
    | NormalInitConfig
    | TruncatedNormalInitConfig
    | SmallInitConfig
    | ZerosInitConfig
    | DiagonalInitConfig
)

ScaleInitConfig = OnesInitConfig | ConstantInitConfig | LinspaceInitConfig | ZerosInitConfig

EmbeddingInitConfig = SmallInitConfig
