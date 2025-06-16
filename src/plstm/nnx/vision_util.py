import collections
import itertools
import math

import einops
from plstm.nnx_dummy import nnx
import jax
import jax.lax as lax
import numpy as np

from ..config.vision_util import VitPatchEmbedConfig, VitPosEmbed2dConfig
from .dtype import str_dtype_to_jax
from .initialization import InitInterface


# adapted from timm (timm/models/layers/helpers.py)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n
            return x
        return tuple(itertools.repeat(x, n))

    return parse


# adapted from timm (timm/models/layers/helpers.py)
def to_ntuple(x, n):
    return _ntuple(n=n)(x)


def interpolate_sincos(embed, seqlens, mode="bicubic"):
    assert embed.ndim - 2 == len(seqlens)
    # JAX doesn't have a direct equivalent to F.interpolate, so we'll use jax.image
    # For now, we only support bicubic interpolation for 2D
    if mode != "bicubic" or len(seqlens) != 2:
        raise NotImplementedError(f"Interpolation mode {mode} not implemented for dimension {len(seqlens)}")

    embed = embed.transpose(0, 3, 1, 2)
    embed = jax.image.resize(embed, shape=(1, embed.shape[1], seqlens[0], seqlens[1]), method="bicubic")
    embed = embed.transpose(0, 2, 3, 1)
    return embed


class SequenceConv2d(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        seqlens: tuple[int, int] | None = None,
        dtype: str = "bfloat16",
        param_dtype: str = "float32",
        *,
        rngs: nnx.Rngs,
    ):
        nnx.Module.__init__(self)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.seqlens = seqlens
        self.dtype = dtype
        self.param_dtype = param_dtype

        scale = 1 / np.sqrt(in_channels * kernel_size[0] * kernel_size[1])
        param_dtype_jax = str_dtype_to_jax(param_dtype)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (out_channels, in_channels // groups, *kernel_size), dtype=param_dtype_jax)
            * scale
        )
        self.bias = (
            nnx.Param(jax.random.normal(rngs.params(), (out_channels,), dtype=param_dtype_jax) * scale)
            if bias
            else None
        )

    def __call__(self, x, deterministic: bool = False):
        assert x.ndim == 3
        if self.seqlens is None:
            # assuming square input
            h = math.sqrt(x.shape[1])
            assert h.is_integer()
            h = int(h)
        else:
            assert len(self.seqlens) == 2
            h = self.seqlens[0]

        dtype = str_dtype_to_jax(self.dtype)
        x = x.astype(dtype)
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=h)
        x = lax.conv_general_dilated(
            x,
            self.weight.astype(dtype),
            window_strides=self.stride,
            padding=[(p, p) for p in self.padding],
            lhs_dilation=(1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=self.groups,
        )
        if self.bias is not None:
            x = x + self.bias.astype(dtype).reshape(1, -1, 1, 1)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        return x


class VitPatchEmbed(nnx.Module):
    def __init__(
        self,
        config: VitPatchEmbedConfig,
        *,
        rngs: nnx.Rngs,
    ):
        nnx.Module.__init__(self)
        self.config = config
        self.ndim = len(config.resolution)
        self.patch_size = to_ntuple(config.patch_size, n=self.ndim)
        if config.stride is None:
            self.stride = self.patch_size
        else:
            self.stride = to_ntuple(config.stride, n=self.ndim)

        for i in range(self.ndim):
            assert (
                config.resolution[i] % self.patch_size[i] == 0
            ), f"resolution[{i}] % patch_size[{i}] != 0 (resolution={config.resolution} patch_size={config.patch_size})"

        self.seqlens = [config.resolution[i] // self.patch_size[i] for i in range(self.ndim)]
        if self.patch_size == self.stride:
            self.num_patches = int(np.prod(self.seqlens))
        else:
            raise NotImplementedError

        # Initialize the appropriate convolution
        self.proj = nnx.Conv(
            in_features=config.num_channels,
            out_features=config.dim,
            kernel_size=self.patch_size,
            kernel_init=config.weight_init.instantiate(InitInterface),
            strides=self.stride,
            rngs=rngs,
            dtype=str_dtype_to_jax(config.dtype),
            param_dtype=str_dtype_to_jax(config.param_dtype),
        )

    def __call__(self, x, deterministic: bool = False):
        # take inputs as feature last
        if self.config.channels_first:
            x = x.transpose(0, 2, 3, 1)
        assert all(
            x.shape[i + 1] % self.patch_size[i] == 0 for i in range(self.ndim)
        ), f"x.shape={x.shape} incompatible with patch_size={self.patch_size}"

        x = self.proj(x)
        return x


class VitPosEmbed2d(nnx.Module):
    def __init__(
        self,
        config: VitPosEmbed2dConfig,
        *,
        rngs: nnx.Rngs,
    ):
        nnx.Module.__init__(self)
        self.config = config
        # Initialize with truncated normal distribution
        param_dtype = str_dtype_to_jax(config.param_dtype)
        init_embed = config.embed_init.instantiate(InitInterface)(
            rngs.params(), shape=(1, *config.seqlens, config.dim), dtype=param_dtype
        )
        self.embed = nnx.Param(init_embed)

    @property
    def _expected_x_ndim(self):
        return len(self.config.seqlens) + 2

    def __call__(self, x, deterministic: bool = False):
        assert x.ndim == self._expected_x_ndim
        dtype = str_dtype_to_jax(self.config.dtype)
        x = x.astype(dtype)
        if x.shape[1:-1] != self.embed.shape[1:-1]:
            assert self.config.allow_interpolation
            embed = interpolate_sincos(embed=self.embed.astype(dtype), seqlens=x.shape[1:-1])
        else:
            embed = self.embed.astype(dtype)
        return x + embed
