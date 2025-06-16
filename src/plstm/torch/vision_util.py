import collections
import itertools
import math

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config.vision_util import VitPatchEmbedConfig, VitPosEmbed2dConfig, DropPathConfig
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


# from kappamodules.functional.pos_embed import interpolate_sincos
def interpolate_sincos(embed, seqlens, mode="bicubic"):
    assert embed.ndim - 2 == len(seqlens)
    embed = F.interpolate(
        einops.rearrange(embed, "1 ... dim -> 1 dim ..."),
        size=seqlens,
        mode=mode,
    )
    embed = einops.rearrange(embed, "1 dim ... -> 1 ... dim")
    return embed


class SequenceConv2d(nn.Conv2d):
    def __init__(self, *args, seqlens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seqlens = seqlens

    def forward(self, x):
        assert x.ndim == 3
        if self.seqlens is None:
            # assuming square input
            h = math.sqrt(x.size(1))
            assert h.is_integer()
            h = int(h)
        else:
            assert len(self.seqlens) == 2
            h = self.seqlens[0]
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=h)
        x = super().forward(x)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        return x


# from kappamodules.vit import VitPatchEmbed
class VitPatchEmbed(nn.Module):
    def __init__(self, config: VitPatchEmbedConfig):
        nn.Module.__init__(self)
        self.config = config
        self.resolution = config.resolution
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
            # use primitive type as np.prod gives np.int which is not compatible with all serialization/logging
            self.num_patches = int(np.prod(self.seqlens))
        else:
            if self.ndim == 1:
                conv_func = F.conv1d
            elif self.ndim == 2:
                conv_func = F.conv2d
            elif self.ndim == 3:
                conv_func = F.conv3d
            else:
                raise NotImplementedError
            self.num_patches = conv_func(
                input=torch.zeros(1, 1, *config.resolution),
                weight=torch.zeros(1, 1, *self.patch_size),
                stride=self.stride,
            ).numel()

        if self.ndim == 1:
            conv_ctor = nn.Conv1d
        elif self.ndim == 2:
            conv_ctor = nn.Conv2d
        elif self.ndim == 3:
            conv_ctor = nn.Conv3d
        else:
            raise NotImplementedError

        self.proj = conv_ctor(config.num_channels, config.dim, kernel_size=self.patch_size, stride=self.stride)
        self.reset_parameters()

    def reset_parameters(self):
        self.config.weight_init.instantiate(InitInterface)(self.proj.weight)
        self.config.bias_init.instantiate(InitInterface)(self.proj.bias)

    def forward(self, x):
        if not self.config.channels_first:
            x = einops.rearrange(x, "b ... c -> b c ...")
        assert all(
            x.size(i + 2) % self.patch_size[i] == 0 for i in range(self.ndim)
        ), f"x.shape={x.shape} incompatible with patch_size={self.patch_size}"
        x = self.proj(x)
        x = einops.rearrange(x, "b c ... -> b ... c")
        return x


# from kappamodules.vit import VitPosEmbed2d
class VitPosEmbed2d(nn.Module):
    def __init__(self, config: VitPosEmbed2dConfig):
        nn.Module.__init__(self)
        self.config = config
        self.seqlens = config.seqlens
        self.embed = nn.Parameter(torch.zeros(1, *config.seqlens, config.dim))
        self.reset_parameters()

    @property
    def _expected_x_ndim(self):
        return len(self.seqlens) + 2

    def reset_parameters(self):
        self.config.embed_init.instantiate(InitInterface)(self.embed)

    def forward(self, x):
        assert x.ndim == self._expected_x_ndim
        if x.shape[1:] != self.embed.shape[1:]:
            assert self.config.allow_interpolation
            embed = interpolate_sincos(embed=self.embed, seqlens=x.shape[1:-1])
        else:
            embed = self.embed
        return x + embed


# does not work in JAX yet
# from kappamodules.layers import DropPath
class DropPath(nn.Sequential):
    """
    Efficiently drop paths (Stochastic Depth) per sample such that dropped samples are not processed.
    This is a subclass of nn.Sequential and can be used either as standalone Module or like nn.Sequential.
    Examples::
        >>> # use as nn.Sequential module
        >>> sequential_droppath = DropPath(nn.Linear(4, 4), config=DropPathConfig(drop_prob=0.2))
        >>> y = sequential_droppath(torch.randn(10, 4))

        >>> # use as standalone module
        >>> standalone_layer = nn.Linear(4, 4)
        >>> standalone_droppath = DropPath(config=DropPathConfig(drop_prob=0.2))
        >>> y = standalone_droppath(torch.randn(10, 4), standalone_layer)
    """

    def __init__(
        self,
        *args,
        config: DropPathConfig,
    ):
        super().__init__(*args)
        assert 0.0 <= config.drop_prob < 1.0
        self._drop_prob = config.drop_prob

    @property
    def drop_prob(self):
        return self._drop_prob

    @drop_prob.setter
    def drop_prob(self, value):
        assert 0.0 <= value < 1.0
        self._drop_prob = value

    @property
    def keep_prob(self):
        return 1.0 - self.drop_prob

    def forward(self, x, residual_path=None, residual_path_kwargs=None):
        assert (len(self) == 0) ^ (residual_path is None)
        residual_path_kwargs = residual_path_kwargs or {}
        if self.drop_prob == 0.0 or not self.training:
            if residual_path is None:
                return x + super().forward(x, **residual_path_kwargs)
            else:
                return x + residual_path(x, **residual_path_kwargs)
        # generate indices to keep (propagated through transform path)
        bs = len(x)
        if self.config.stochastic_drop_prob:
            perm = torch.empty(bs, device=x.device).bernoulli_(self.keep_prob).nonzero().squeeze(1)
            scale = 1 / self.keep_prob
        else:
            keep_count = max(int(bs * self.keep_prob), 1)
            scale = bs / keep_count
            perm = torch.randperm(bs, device=x.device)[:keep_count]

        # propagate
        if self.config.scale_by_keep:
            alpha = scale
        else:
            alpha = 1.0
        # reduce kwargs (e.g. used for DiT block where scale/shift/gate is passed and also has to be reduced)
        residual_path_kwargs = {
            key: value[perm] if torch.is_tensor(value) else value for key, value in residual_path_kwargs.items()
        }
        if residual_path is None:
            residual = super().forward(x[perm], **residual_path_kwargs)
        else:
            residual = residual_path(x[perm], **residual_path_kwargs)
        return torch.index_add(
            x.flatten(start_dim=1),
            dim=0,
            index=perm,
            source=residual.to(x.dtype).flatten(start_dim=1),
            alpha=alpha,
        ).view_as(x)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
