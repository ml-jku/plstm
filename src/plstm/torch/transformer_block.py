import torch
import torch.nn as nn
import torch.nn.functional as F

from compoconf import register
from ..config.transformer_block import TransformerBlockConfig
from ..config.initialization import ZerosInitConfig
from .interfaces import ResidualModule
from .norm import NormInterface
from .scale import ScaleLayer
from .initialization import InitInterface


@register
class TransformerBlock(ResidualModule):
    config: TransformerBlockConfig

    def __init__(self, config: TransformerBlockConfig):
        ResidualModule.__init__(self, config)
        self.config = config

        self.norm = config.norm.instantiate(NormInterface)
        self.norm2 = config.norm.instantiate(NormInterface)
        self.scale = ScaleLayer(config.scale)
        self.scale2 = ScaleLayer(config.scale)
        self.drop_path = nn.Dropout(config.drop_path_rate)

        self.qkv = nn.Linear(
            self.config.input_dim,
            3 * self.config.input_dim,
            bias=self.config.bias,
        )
        self.outproj = nn.Linear(
            self.config.input_dim,
            self.config.input_dim,
            bias=self.config.bias,
        )

        if not self.config.gated:
            self.upproj = nn.Linear(
                self.config.input_dim,
                self.config.inner_input_dim,
                bias=self.config.bias,
            )
        else:
            self.upproj = nn.Linear(
                self.config.input_dim,
                2 * self.config.inner_input_dim,
                bias=self.config.bias,
            )

        self.downproj = nn.Linear(
            self.config.inner_input_dim,
            self.config.input_dim,
            bias=self.config.bias,
        )

        # Initialize weights similar to NNX implementation
        self.reset_parameters()

    def reset_parameters(self):
        self.config.attn_weight_init.instantiate(InitInterface)(self.qkv.weight)
        self.config.out_weight_init.instantiate(InitInterface)(self.outproj.weight)

        self.config.upproj_weight_init.instantiate(InitInterface)(self.upproj.weight)
        self.config.downproj_weight_init.instantiate(InitInterface)(self.downproj.weight)
        if self.config.bias:
            ZerosInitConfig().instantiate(InitInterface)(self.outproj.bias)
            ZerosInitConfig().instantiate(InitInterface)(self.qkv.bias)
            self.config.upproj_bias_init.instantiate(InitInterface)(self.upproj.bias)
            self.config.downproj_bias_init.instantiate(InitInterface)(self.downproj.bias)

    def _gating_function(self, x):
        if self.config.gating_function == "silu":
            return F.silu(x)
        elif self.config.gating_function == "gelu":
            return F.gelu(x)
        else:
            raise ValueError("Bad gating function")

    def forward(self, x, deterministic: bool = False):
        # First normalization and multi-head attention
        x0 = self.norm(x)
        x_shape = x0.shape
        x1 = x0.reshape(x_shape[0], -1, x_shape[-1])
        qkv = self.qkv(x1)
        q, k, v = torch.split(qkv, (x_shape[-1], x_shape[-1], x_shape[-1]), dim=-1)
        # return q
        z1 = torch.nn.functional.scaled_dot_product_attention(
            q.reshape(x_shape[0], -1, self.config.num_heads, x_shape[-1] // self.config.num_heads).transpose(-2, -3),
            k.reshape(x_shape[0], -1, self.config.num_heads, x_shape[-1] // self.config.num_heads).transpose(-2, -3),
            v.reshape(x_shape[0], -1, self.config.num_heads, x_shape[-1] // self.config.num_heads).transpose(-2, -3),
        )
        y1 = z1.transpose(-2, -3).reshape(x_shape)
        y1 = self.outproj(y1)

        # Residual connection with optional drop path
        if self.config.skip:
            x2 = (
                self.scale(y1) * self.drop_path(y1.new_ones((y1.shape[0],))).view((-1, *[1] * (y1.ndim - 1))) + x
                if not deterministic
                else self.scale(y1) + x
            )
        else:
            x2 = self.scale(y1)

        # Second normalization and projection
        x3 = self.norm2(x2)
        x4 = self.upproj(x3)

        # Gating mechanism
        if self.config.gated:
            x5, x6 = torch.split(x4, [self.config.inner_input_dim, self.config.inner_input_dim], dim=-1)
            y2 = x5 * self._gating_function(x6)
        else:
            y2 = self._gating_function(x4)

        # Final projection and residual connection
        y3 = self.scale2(self.downproj(y2))

        if self.config.skip:
            y4 = (
                y3 * self.drop_path(y3.new_ones((y3.shape[0],))).view((-1, *[1] * (y3.ndim - 1))) + x2
                if not deterministic
                else y3 + x2
            )
        else:
            y4 = y3

        return y4
