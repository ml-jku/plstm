import collections
import itertools
import math

import einops
from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from ..config.vision_util import VitPatchEmbedConfig, VitPosEmbed2dConfig, PatchEmbedConfig
from .dtype import str_dtype_to_jax


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

    embed = einops.rearrange(embed, "1 ... dim -> 1 dim ...")
    embed = jax.image.resize(embed, shape=(1, embed.shape[1], seqlens[0], seqlens[1]), method="bicubic")
    embed = einops.rearrange(embed, "1 dim ... -> 1 ... dim")
    return embed


def fixed_sincos2d_embeddings(ncols: int, nrows: int, dim: int) -> jnp.ndarray:
    freqs = 1 / (10000 ** jnp.linspace(0, 1, dim // 4))
    x = jnp.outer(jnp.arange(0, nrows, dtype=jnp.float32), freqs)
    y = jnp.outer(jnp.arange(0, ncols, dtype=jnp.float32), freqs)

    x = jnp.broadcast_to(x[None, :, :], (ncols, nrows, dim // 4))
    y = jnp.broadcast_to(y[:, None, :], (ncols, nrows, dim // 4))
    return jnp.concatenate((jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)), axis=2)


class SequenceConv2d(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] = 0
    dilation: int | tuple[int, int] = 1
    groups: int = 1
    bias: bool = True
    seqlens: tuple[int, int] | None = None
    dtype: str = "bfloat16"
    param_dtype: str = "float32"

    def setup(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size

        if isinstance(self.stride, int):
            self.stride_tuple = (self.stride, self.stride)
        else:
            self.stride_tuple = self.stride

        if isinstance(self.padding, int):
            self.padding_tuple = (self.padding, self.padding)
        else:
            self.padding_tuple = self.padding

        if isinstance(self.dilation, int):
            self.dilation_tuple = (self.dilation, self.dilation)
        else:
            self.dilation_tuple = self.dilation

        scale = 1 / np.sqrt(self.in_channels * kernel_size[0] * kernel_size[1])
        param_dtype_jax = str_dtype_to_jax(self.param_dtype)

        self.weight = self.param(
            "weight",
            lambda key, shape, dtype: jax.random.normal(key, shape, dtype) * scale,
            (self.out_channels, self.in_channels // self.groups, *kernel_size),
            param_dtype_jax,
        )

        if self.bias:
            self.bias_param = self.param(
                "bias",
                lambda key, shape, dtype: jax.random.normal(key, shape, dtype) * scale,
                (self.out_channels,),
                param_dtype_jax,
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
            window_strides=self.stride_tuple,
            padding=[(p, p) for p in self.padding_tuple],
            lhs_dilation=(1, 1),
            rhs_dilation=self.dilation_tuple,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=self.groups,
        )
        if self.bias:
            x = x + self.bias_param.astype(dtype).reshape(1, -1, 1, 1)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        return x


class VitPatchEmbed(nn.Module):
    config: VitPatchEmbedConfig

    def setup(self):
        self.ndim = len(self.config.resolution)
        self.patch_size = to_ntuple(self.config.patch_size, n=self.ndim)
        if self.config.stride is None:
            self.stride = self.patch_size
        else:
            self.stride = to_ntuple(self.config.stride, n=self.ndim)

        for i in range(self.ndim):
            assert self.config.resolution[i] % self.patch_size[i] == 0, (
                f"resolution[{i}] % patch_size[{i}] != 0 (resolution={self.config.resolution} "
                f"patch_size={self.config.patch_size})"
            )

        self.seqlens = [self.config.resolution[i] // self.patch_size[i] for i in range(self.ndim)]
        if self.patch_size == self.stride:
            self.num_patches = int(np.prod(self.seqlens))
        else:
            raise NotImplementedError

        # Initialize the appropriate convolution
        self.proj = nn.Conv(
            features=self.config.dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            dtype=str_dtype_to_jax(self.config.dtype),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
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


class VitPosEmbed2d(nn.Module):
    config: VitPosEmbed2dConfig

    def setup(self):
        # Initialize with truncated normal distribution
        param_dtype = str_dtype_to_jax(self.config.param_dtype)
        self.embed = self.param(
            "embed",
            lambda key, shape, dtype: jax.random.truncated_normal(key, -2, 2, shape, dtype) * 0.02,
            (1, *self.config.seqlens, self.config.dim),
            param_dtype,
        )

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


class PatchEmbed(nn.Module):
    config: PatchEmbedConfig

    def setup(self):
        self.wte = nn.Conv(
            self.config.dim,
            kernel_size=(self.config.patch_size, self.config.patch_size),
            strides=(self.config.patch_size, self.config.patch_size),
            padding="VALID",
        )

        # Calculate number of patches based on input shape and patch size
        # This is a placeholder - in a real implementation, you would need to know the input shape
        # For now, we'll just use a default value, it

        if self.config.pos_emb == "learnable":
            self.wpe = self.param(
                "wpe", nn.initializers.truncated_normal(stddev=0.02), (1, *self.config.seqlens, self.config.output_dim)
            )
        elif self.config.pos_emb == "sincos2d":
            self.wpe = fixed_sincos2d_embeddings(self.seqlens[0], self.seqlens[1], self.config.output_dim)
        elif self.config.pos_emb == "none":
            self.wpe = 0.0

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        x = self.wte(x)

        # Add positional embeddings if not "none"
        if self.config.pos_emb != "none":
            if x.shape[1:-1] != self.wpe.shape[1:-1]:
                assert self.config.allow_interpolation
                embed = interpolate_sincos(embed=self.wpe, seqlens=x.shape[1:-1])
            else:
                embed = self.wpe
            x = x + embed

        # Reshape to sequence format
        if self.config.flatten:
            x = x.reshape(x.shape[0], -1, self.config.output_dim)

        return x
