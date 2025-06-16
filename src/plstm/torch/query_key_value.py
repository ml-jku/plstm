import torch
import torch.nn as nn

from ..config.query_key_value import (
    QueryLayerConfig,
    KeyLayerConfig,
    ValueLayerConfig,
)
from .dtype import str_dtype_to_torch
from .initialization import InitInterface


class QueryLayer(nn.Module):
    config: QueryLayerConfig

    def __init__(self, config: QueryLayerConfig):
        super().__init__()
        self.config = config

        # Initialize weight
        weight_shape = [
            max(1, config.num_heads // config.sub_heads),
            config.DK // max(1, config.sub_heads // config.num_heads),
            config.JQ,
            config.sub_heads,
            config.input_dim // config.sub_heads,
        ]

        self.weight = nn.Parameter(torch.zeros(weight_shape, dtype=str_dtype_to_torch(config.param_dtype)))

        # Initialize bias if needed
        if self.config.bias:
            bias_shape = [config.num_heads, config.DK, config.JQ]
            self.bias = nn.Parameter(torch.zeros(bias_shape, dtype=str_dtype_to_torch(config.param_dtype)))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        config = self.config

        config.weight_init.instantiate(InitInterface)(self.weight)
        if config.bias:
            config.bias_init.instantiate(InitInterface)(self.bias)

    def _normalization(self, x):
        if self.config.normalization == "none":
            return x
        elif self.config.normalization == "l2":
            x = x.to(dtype=str_dtype_to_torch(self.config.dtype))
            return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)

    def forward(self, x):
        x_reshaped = x.reshape(
            *x.shape[:-1],
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        )

        # Cast to computation dtype for mixed precision
        weight = self.weight.to(dtype=str_dtype_to_torch(self.config.dtype))
        x_reshaped = x_reshaped.to(dtype=str_dtype_to_torch(self.config.dtype))

        # Perform einsum-like operation
        out = torch.einsum("hkjsx,...sx->...hskj", weight, x_reshaped)
        out = out.reshape(list(x.shape[:-1]) + [self.config.num_heads, self.config.DK, self.config.JQ])

        if self.config.bias:
            bias = self.bias.to(dtype=str_dtype_to_torch(self.config.dtype))
            out = out + bias

        return self._normalization(out)

    def extra_repr(self):
        return (
            f"heads={self.config.num_heads}, subheads={self.config.sub_heads},"
            f" input_dim={self.config.input_dim}, JQ={self.config.JQ}, DK={self.config.DK})"
        )


class KeyLayer(nn.Module):
    config: KeyLayerConfig

    def __init__(self, config: KeyLayerConfig):
        super().__init__()
        self.config = config

        # Initialize weight
        weight_shape = [
            max(1, config.num_heads // config.sub_heads),
            config.DK // max(1, config.sub_heads // config.num_heads),
            config.JK,
            config.sub_heads,
            config.input_dim // config.sub_heads,
        ]

        self.weight = nn.Parameter(torch.zeros(weight_shape, dtype=str_dtype_to_torch(config.param_dtype)))

        # Initialize bias if needed
        if self.config.bias:
            bias_shape = [config.num_heads, config.DK, config.JK]
            self.bias = nn.Parameter(torch.zeros(bias_shape, dtype=str_dtype_to_torch(config.param_dtype)))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        config = self.config

        config.weight_init.instantiate(InitInterface)(self.weight)
        if config.bias:
            config.bias_init.instantiate(InitInterface)(self.bias)

    def _normalization(self, x):
        if self.config.normalization == "none":
            return x
        elif self.config.normalization == "l2":
            x = x.to(dtype=str_dtype_to_torch(self.config.dtype))
            return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)

    def forward(self, x):
        x_reshaped = x.reshape(
            *x.shape[:-1],
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        )

        # Cast to computation dtype for mixed precision
        weight = self.weight.to(dtype=str_dtype_to_torch(self.config.dtype))
        x_reshaped = x_reshaped.to(dtype=str_dtype_to_torch(self.config.dtype))

        # Perform einsum-like operation
        out = torch.einsum("hkjsx,...sx->...hskj", weight, x_reshaped)
        out = out.reshape(list(x.shape[:-1]) + [self.config.num_heads, self.config.DK, self.config.JK])

        if self.config.bias:
            bias = self.bias.to(dtype=str_dtype_to_torch(self.config.dtype))
            out = out + bias

        return self._normalization(out)

    def extra_repr(self):
        return (
            f"heads={self.config.num_heads}, subheads={self.config.sub_heads},"
            f" input_dim={self.config.input_dim}, JK={self.config.JK}, DK={self.config.DK})"
        )


class ValueLayer(nn.Module):
    config: ValueLayerConfig

    def __init__(self, config: ValueLayerConfig):
        super().__init__()
        self.config = config

        # Initialize weight
        weight_shape = [
            max(1, config.num_heads // config.sub_heads),
            config.DV // max(1, config.sub_heads // config.num_heads),
            config.JV,
            config.sub_heads,
            config.input_dim // config.sub_heads,
        ]

        self.weight = nn.Parameter(torch.zeros(weight_shape, dtype=str_dtype_to_torch(config.param_dtype)))

        # Initialize bias if needed
        if self.config.bias:
            bias_shape = [config.num_heads, config.DV, config.JV]
            self.bias = nn.Parameter(torch.zeros(bias_shape, dtype=str_dtype_to_torch(config.param_dtype)))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        config = self.config

        config.weight_init.instantiate(InitInterface)(self.weight)
        if config.bias:
            config.bias_init.instantiate(InitInterface)(self.bias)

    def _normalization(self, x):
        if self.config.normalization == "none":
            return x
        elif self.config.normalization == "l2":
            x = x.to(dtype=str_dtype_to_torch(self.config.dtype))
            return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)

    def forward(self, x):
        x_reshaped = x.reshape(
            *x.shape[:-1],
            self.config.sub_heads,
            self.config.input_dim // self.config.sub_heads,
        )

        # Cast to computation dtype for mixed precision
        weight = self.weight.to(dtype=str_dtype_to_torch(self.config.dtype))
        x_reshaped = x_reshaped.to(dtype=str_dtype_to_torch(self.config.dtype))

        # Perform einsum-like operation
        out = torch.einsum("hkjsx,...sx->...hskj", weight, x_reshaped)
        out = out.reshape(list(x.shape[:-1]) + [self.config.num_heads, self.config.DV, self.config.JV])

        if self.config.bias:
            bias = self.bias.to(dtype=str_dtype_to_torch(self.config.dtype))
            out = out + bias

        return self._normalization(out)

    def extra_repr(self):
        return (
            f"heads={self.config.num_heads}, subheads={self.config.sub_heads},"
            f" input_dim={self.config.input_dim}, JV={self.config.JV}, DV={self.config.DV})"
        )
