from dataclasses import dataclass, field
from typing import Literal, Any

from .norm import NormConfig, RMSNormConfig
from .vision_blocks import pLSTMVisionBlockConfig1
from .block_stack import BlockStackConfig
from .vision_util import VitPatchEmbedConfig, VitPosEmbed2dConfig, PatchEmbedConfig
from .scalar import SoftCapFunctionConfig
from compoconf import ConfigInterface
from .dtype import DTYPES
from .initialization import BiasInitConfig, WeightInitConfig, ZerosInitConfig, SmallInitConfig


@dataclass
class pLSTMVisionModelConfig(ConfigInterface):
    _short_name: str = "pLSTMVisionModel"
    dim: int = 128
    num_channels: int = 3
    patch_size: int = 16
    num_blocks: int = 12
    output_shape: tuple[int, ...] = (1000,)
    mode: str = "classifier"
    pooling: Literal["corners", "center", "cls", "sum", "mean"] = "corners"
    stride: int | None = None
    norm_bias: bool = True
    num_heads: int = 8
    seqlens: list[int] | None = None
    resolution: tuple[int, int] | None = (224, 224)
    channels_first: bool = False
    input_shape: tuple[int, int, int] | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    head_init_std: float = 0.02  # deprecated
    num_patches: int | None = None

    norm: NormConfig | None = None
    block_stack: BlockStackConfig | None = None

    patch_embed: VitPatchEmbedConfig | None = None
    use_pos_embed: bool = True
    pos_embed: VitPosEmbed2dConfig | None = None

    logit_softcap: SoftCapFunctionConfig | None = None

    head_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    head_bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        if self.input_shape is None:
            self.input_shape = (
                (self.num_channels, *self.resolution) if self.channels_first else (*self.resolution, self.num_channels)
            )
        if self.resolution is None:
            if self.channels_first:
                self.resolution = self.input_shape[1:]
            else:
                self.resolution = self.input_shape[:-1]
        if self.seqlens is None:
            self.seqlens = [self.resolution[i] // self.patch_size for i in range(len(self.resolution))]
        assert self.seqlens == [self.resolution[i] // self.patch_size for i in range(len(self.resolution))]
        if self.num_patches is None:
            self.num_patches = self.seqlens[0] * self.seqlens[1]
        assert self.num_patches == self.seqlens[0] * self.seqlens[1]
        assert self.num_heads > 0
        assert self.dim % self.num_heads == 0
        assert self.resolution[0] % self.patch_size == 0 and self.resolution[1] % self.patch_size == 0

        if self.patch_embed is None:
            self.patch_embed = VitPatchEmbedConfig(
                patch_size=self.patch_size,
                num_channels=self.num_channels,
                resolution=self.resolution,
                channels_first=self.channels_first,
                stride=self.stride,
                dim=self.dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        if self.pos_embed is None:
            self.pos_embed = VitPosEmbed2dConfig(
                seqlens=self.seqlens,
                dim=self.dim,
                allow_interpolation=True,
            )

        if self.block_stack is None:
            self.block_stack = BlockStackConfig(
                block=pLSTMVisionBlockConfig1(
                    input_dim=self.dim,
                    num_heads=self.num_heads,
                ),
                input_dim=self.dim,
                num_blocks=self.num_blocks,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        if self.norm is None:
            self.norm = RMSNormConfig(
                input_dim=self.dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )


@dataclass
class pLSTMVisionModelConfig2(ConfigInterface):
    _short_name: str = "pLSTMVisionModel"
    num_channels: int = 3
    num_patches: int | None = None
    patch_size: int = 16
    output_shape: tuple[int, ...] = (1000,)
    mode: str = "classifier"
    pooling: Literal["corners", "center"] = "corners"
    stride: int | None = None
    seqlens: list[int] | None = None
    resolution: tuple[int, int] | None = (224, 244)
    channels_first: bool = False
    input_shape: tuple[int, int, int] | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    norm: NormConfig | None = None
    block_stack: BlockStackConfig | None = None

    patch_embed: PatchEmbedConfig | None = None

    logit_softcap: SoftCapFunctionConfig | None = None

    head_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    head_bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        self.input_shape = (
            (self.num_channels, *self.resolution) if self.channels_first else (*self.resolution, self.num_channels)
        )
        if self.seqlens is None:
            self.seqlens = [self.resolution[i] // self.patch_size for i in range(len(self.resolution))]
        assert self.seqlens == [self.resolution[i] // self.patch_size for i in range(len(self.resolution))]
        if self.num_patches is None:
            self.num_patches = self.seqlens[0] * self.seqlens[1]
        assert self.num_patches == self.seqlens[0] * self.seqlens[1]
        assert self.num_heads > 0
        assert self.dim % self.num_heads == 0

        if self.patch_embed is None:
            self.patch_embed = PatchEmbedConfig(
                patch_size=self.patch_size,
                num_channels=self.num_channels,
                resolution=self.resolution,
                channels_first=self.channels_first,
                dim=self.dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                seqlens=self.seqlens,
                allow_interpolation=True,
            )

        if self.block_stack is None:
            self.block_stack = BlockStackConfig(
                block=pLSTMVisionBlockConfig1(
                    input_dim=self.dim,
                    num_heads=self.num_heads,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                input_dim=self.dim,
                num_blocks=self.num_blocks,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        if self.norm is None:
            self.norm = RMSNormConfig(
                input_dim=self.dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
