import torch
from ..config.source_mark_layer import (
    SourceLayerConfig,
    MarkLayerConfig,
    DirectLayerConfig,
)
from .initialization import InitInterface


class SourceLayer(torch.nn.Module):
    config: SourceLayerConfig

    def __init__(self, config: SourceLayerConfig):
        torch.nn.Module.__init__(self)
        self.config = config

        # Declare parameters
        self.bias = torch.nn.Parameter(torch.zeros([config.num_heads, config.JT, config.JK, config.JV]))

        if self.config.weight:
            weight_shape = [
                config.num_heads // config.sub_heads,
                config.sub_heads,
                config.JT,
                config.JK,
                config.JV,
                config.input_dim // config.sub_heads,
            ]
            self.weight = torch.nn.Parameter(torch.zeros(weight_shape))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.config.bias_init.instantiate(InitInterface)(self.bias)
        # Initialize weight if needed
        if self.config.weight:
            self.config.weight_init.instantiate(InitInterface)(self.weight)

    def _activation(self, x):
        if self.config.activation == "identity":
            return x
        elif self.config.activation == "tanh":
            return self.config.activation_scale * torch.tanh(x / self.config.activation_scale)
        elif self.config.activation == "logsigmoid":
            return torch.exp(self.config.activation_scale * torch.nn.functional.logsigmoid(x))
        else:
            raise ValueError("Bad source activation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input for sub_heads
        x = x.reshape(*x.shape[:-1], self.config.sub_heads, self.config.input_dim // self.config.sub_heads)

        if self.config.weight:
            # Use einsum for the weight multiplication
            result = self.bias + torch.einsum("hsijkd,...sd->...hsijk", self.weight, x).reshape(
                *x.shape[:-2], self.config.num_heads, self.config.JT, self.config.JK, self.config.JV
            )
        else:
            # Just use the bias
            result = self.bias[None, :].unflatten(0, [1 for _ in x.shape[:-2]]).tile(list(x.shape[:-2]) + [1, 1, 1, 1])

        return self._activation(result)


class MarkLayer(torch.nn.Module):
    config: MarkLayerConfig

    def __init__(self, config: MarkLayerConfig):
        torch.nn.Module.__init__(self)
        self.config = config

        # Declare parameters
        self.bias = torch.nn.Parameter(torch.zeros([config.num_heads, config.JO, config.JQ, config.JT]))

        if self.config.weight:
            weight_shape = [
                config.num_heads // config.sub_heads,
                config.sub_heads,
                config.JO,
                config.JQ,
                config.JT,
                config.input_dim // config.sub_heads,
            ]
            self.weight = torch.nn.Parameter(torch.zeros(weight_shape))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.config.bias_init.instantiate(InitInterface)(self.bias)
        # Initialize weight if needed
        if self.config.weight:
            self.config.weight_init.instantiate(InitInterface)(self.weight)

    def _activation(self, x):
        if self.config.activation == "identity":
            return x
        elif self.config.activation == "tanh":
            return self.config.activation_scale * torch.tanh(x / self.config.activation_scale)
        elif self.config.activation == "logsigmoid":
            return torch.exp(self.config.activation_scale * torch.nn.functional.logsigmoid(x))
        else:
            raise ValueError("Bad source activation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input for sub_heads
        x = x.reshape(*x.shape[:-1], self.config.sub_heads, self.config.input_dim // self.config.sub_heads)

        if self.config.weight:
            # Use einsum for the weight multiplication
            result = self.bias + torch.einsum("hsijkd,...sd->...hsijk", self.weight, x).reshape(
                *x.shape[:-2], self.config.num_heads, self.config.JO, self.config.JQ, self.config.JT
            )
        else:
            # Just use the bias
            result = self.bias[None, :].unflatten(0, [1 for _ in x.shape[:-2]]).tile(list(x.shape[:-2]) + [1, 1, 1, 1])

        return self._activation(result)


class DirectLayer(torch.nn.Module):
    config: DirectLayerConfig

    def __init__(self, config: DirectLayerConfig):
        torch.nn.Module.__init__(self)
        self.config = config

        # Declare parameters
        self.bias = torch.nn.Parameter(torch.zeros([config.num_heads, config.JO, config.JQ, config.JK, config.JV]))

        if self.config.weight:
            weight_shape = [
                config.num_heads // config.sub_heads,
                config.sub_heads,
                config.JO,
                config.JQ,
                config.JK,
                config.JV,
                config.input_dim // config.sub_heads,
            ]
            self.weight = torch.nn.Parameter(torch.zeros(weight_shape))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.config.bias_init.instantiate(InitInterface)(self.bias)
        # Initialize weight if needed
        if self.config.weight:
            self.config.weight_init.instantiate(InitInterface)(self.weight)

    def _activation(self, x):
        if self.config.activation == "identity":
            return x
        elif self.config.activation == "tanh":
            return self.config.activation_scale * torch.tanh(x / self.config.activation_scale)
        elif self.config.activation == "logsigmoid":
            return torch.exp(self.config.activation_scale * torch.nn.functional.logsigmoid(x))
        else:
            raise ValueError("Bad source activation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input for sub_heads
        x = x.reshape(*x.shape[:-1], self.config.sub_heads, self.config.input_dim // self.config.sub_heads)

        if self.config.weight:
            # Use einsum for the weight multiplication
            result = self.bias + torch.einsum("hsijkld,...sd->...hsijkl", self.weight, x).reshape(
                *x.shape[:-2], self.config.num_heads, self.config.JO, self.config.JQ, self.config.JK, self.config.JV
            )
        else:
            # Just use the bias
            result = (
                self.bias[None, :].unflatten(0, [1 for _ in x.shape[:-2]]).tile(list(x.shape[:-2]) + [1, 1, 1, 1, 1])
            )

        return self._activation(result)
