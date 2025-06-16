import torch
from torch import nn

from ..config.convolution import Convolution1DLayerConfig, Convolution2DLayerConfig


class Convolution1DLayer(nn.Module):
    config: Convolution1DLayerConfig

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config

        self._conv = nn.Conv1d(
            in_channels=self.config.input_dim,
            out_channels=self.config.input_dim,
            kernel_size=self.config.kernel_size,
            padding=self.config.kernel_size - 1,
            groups=self.config.input_dim,
            bias=self.config.bias,
        )

    def reset_parameters(self, *args, **kwargs):
        self._conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        if len(x_shape) == 4:
            B, T, H, D = x_shape
            x = x.view(B, T, H * D)

        return self._conv(x.transpose(2, 1))[:, :, : -(self.config.kernel_size - 1)].transpose(1, 2).reshape(x_shape)


class Convolution2DLayer(nn.Module):
    config: Convolution2DLayerConfig

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self._conv = nn.Conv2d(
            in_channels=self.config.input_dim,
            out_channels=self.config.input_dim,
            kernel_size=self.config.kernel_size,
            padding=(self.config.kernel_size[0] - 1, self.config.kernel_size[1] - 1),
            groups=self.config.input_dim,
            bias=self.config.bias,
        )

    def reset_parameters(self, *args, **kwargs):
        self._conv.reset_parameters()

    def forward(self, x: torch.Tensor):
        x_shape = x.shape
        if len(x_shape) == 5:
            B, X, Y, H, D = x_shape
            x = x.view(B, X, Y, H * D)
        return (
            self._conv(x.permute(0, 3, 1, 2))[
                :,
                :,
                : -(self.config.kernel_size[0] - 1),
                : -(self.config.kernel_size[1] - 1),
            ]
            .permute(0, 2, 3, 1)
            .reshape(x_shape)
        )
