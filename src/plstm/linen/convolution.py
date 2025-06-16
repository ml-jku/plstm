import jax
from flax import linen as nn

from ..config.convolution import Convolution1DLayerConfig, Convolution2DLayerConfig
from .dtype import str_dtype_to_jax


class Convolution1DLayer(nn.Module):
    """1D Convolution layer with grouped convolutions."""

    config: Convolution1DLayerConfig

    def setup(self):
        self._conv = nn.Conv(
            features=self.config.input_dim,
            kernel_size=(self.config.kernel_size,),
            padding=((self.config.kernel_size - 1, 0),),
            feature_group_count=self.config.input_dim,
            use_bias=self.config.bias,
            dtype=str_dtype_to_jax(self.config.dtype),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x_shape = x.shape
        if len(x_shape) == 4:
            B, T, H, D = x_shape
            x = x.reshape(B, T, H * D)

        y = self._conv(x)  # [:, : -self.config.kernel_size + 1, :]

        return y.reshape(x_shape)


class Convolution2DLayer(nn.Module):
    """2D Convolution layer with grouped convolutions."""

    config: Convolution2DLayerConfig

    def setup(self):
        self._conv = nn.Conv(
            features=self.config.input_dim,
            kernel_size=self.config.kernel_size,
            padding=((self.config.kernel_size[0] - 1, 0), (self.config.kernel_size[1] - 1, 0)),
            feature_group_count=self.config.input_dim,
            use_bias=self.config.bias,
            dtype=str_dtype_to_jax(self.config.dtype),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x_shape = x.shape
        if len(x_shape) == 5:
            B, X, Y, H, D = x_shape
            x = x.reshape(B, X, Y, H * D)

        # Apply convolution and remove padding
        x = self._conv(x)

        # x = x[:, : -self.config.kernel_size[0] + 1, : -self.config.kernel_size[1] + 1]
        return x.reshape(x_shape)
