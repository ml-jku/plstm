from dataclasses import dataclass, field
from typing import Literal
from .dtype import DTYPES
from ..util import assert_check_literals
from .initialization import WeightInitConfig, BiasInitConfig, SmallInitConfig, ZerosInitConfig


@dataclass
class VitPatchEmbedConfig:
    dim: int = -1
    num_channels: int = -1
    resolution: tuple[int, ...] = None
    channels_first: bool = False
    patch_size: int | list[int] = None
    stride: int | list[int] | None = None
    weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)

    param_dtype: DTYPES = "float32"
    dtype: DTYPES = "bfloat16"

    def __post_init__(self):
        assert self.dim > 0, "dim must be positive"
        assert self.num_channels > 0, "num_channels must be positive"
        assert self.resolution is not None, "resolution must be specified"
        assert self.patch_size is not None, "patch_size must be specified"
        if isinstance(self.resolution, (list, tuple)):
            assert all(r > 0 for r in self.resolution), "all resolution dimensions must be positive"
        else:
            raise ValueError("resolution must be a sequence")
        assert_check_literals(self)


@dataclass
class VitPosEmbed2dConfig:
    seqlens: list[int] = None
    dim: int = -1
    allow_interpolation: bool = True
    learnable: bool = True

    embed_init: WeightInitConfig = field(default_factory=SmallInitConfig)

    param_dtype: DTYPES = "float32"
    dtype: DTYPES = "bfloat16"

    def __post_init__(self):
        assert self.dim > 0, "dim must be positive"
        assert self.seqlens is not None, "seqlens must be specified"
        if isinstance(self.seqlens, (list, tuple)):
            assert all(s > 0 for s in self.seqlens), "all seqlens dimensions must be positive"
        else:
            raise ValueError("seqlens must be a sequence")
        assert_check_literals(self)


@dataclass
class DropPathConfig:
    drop_prob: float = 0.0
    scale_by_keep: bool = True
    stochastic_drop_prob: bool = False

    param_dtype: DTYPES = "float32"
    dtype: DTYPES = "bfloat16"

    def __post_init__(self):
        assert 0.0 <= self.drop_prob < 1.0, "drop_prob must be in [0, 1)"
        assert_check_literals(self)


@dataclass
class PatchEmbedConfig:
    seqlens: tuple[int, ...] | None = None
    allow_interpolation: bool = True
    num_channels: int = 3

    resolution: tuple[int, ...] = field(default_factory=list)
    channels_first: bool = False
    patch_size: int | tuple[int, ...] = (4, 4)

    flatten: bool = False
    output_dim: int = -1
    pos_emb: Literal["learnable", "sincos2d", "none"] = "learnable"
    embed_init: WeightInitConfig = field(default_factory=SmallInitConfig)

    def __post_init__(self):
        assert self.output_dim > 0
        assert_check_literals(self)

        assert len(self.resolution) > 0

        if self.seqlens is None:
            self.seqlens = [res // pat for res, pat in zip(self.resolution, self.patch_size)]
