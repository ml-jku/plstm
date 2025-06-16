from plstm.nnx_dummy import nnx
from flax import linen as nn
import jax
import jax.numpy as jnp
from .dtype import str_dtype_to_jax
from compoconf import register_interface, RegistrableConfigInterface, register
from abc import abstractmethod

from ..config.norm import (
    MultiHeadLayerNormConfig,
    LayerNormConfig,
    MultiHeadRMSNormConfig,
    RMSNormConfig,
    IdentityConfig,
)
from .initialization import InitInterface


def vmap_module(target, axis, num_heads):
    return nn.vmap(
        target=target,
        variable_axes={"params": 0},
        in_axes=axis if axis is not None else -2,
        out_axes=axis if axis is not None else -2,
        axis_size=num_heads,
        split_rngs={"params": True},
    )


@register_interface
class NormInterface(nnx.Module, RegistrableConfigInterface):
    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError


@register
class MultiHeadLayerNorm(NormInterface):
    config: MultiHeadLayerNormConfig

    def __init__(self, config: MultiHeadLayerNormConfig, rngs: nnx.Rngs):
        """Create a multi-head norm layer.

        Effectively vmaps a norm layer over the specified axis.

        Args:
            config: MultiHeadNormLayerConfig for the layer
            rngs:   Random Number Generator Iterator for nnx.
        """
        self.config = config
        norm_class, norm_kwargs = (
            nn.LayerNorm,
            {
                "epsilon": config.eps,
                "use_bias": config.bias,
                "use_scale": config.scale,
                "dtype": str_dtype_to_jax(config.dtype),
                "param_dtype": str_dtype_to_jax(config.param_dtype),
            },
        )

        self.norm = nnx.bridge.ToNNX(vmap_module(norm_class, config.axis, config.num_heads)(**norm_kwargs), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.config.axis is not None:
            return self.norm(x).astype(x.dtype)
        else:
            return (
                self.norm(
                    x.reshape(*x.shape[:-1], self.config.num_heads, self.config.input_dim // self.config.num_heads)
                )
                .reshape(*x.shape)
                .astype(x.dtype)
            )


@register
class MultiHeadRMSNorm(NormInterface):
    config: MultiHeadRMSNormConfig

    def __init__(self, config: MultiHeadRMSNormConfig, rngs: nnx.Rngs):
        """Create a multi-head norm layer.

        Effectively vmaps a norm layer over the specified axis.

        Args:
            config: MultiHeadRMSNormConfig for the layer
            rngs:   Random Number Generator Iterator for nnx.
        """
        self.config = config
        norm_class, norm_kwargs = (
            nn.RMSNorm,
            {
                "epsilon": config.eps,
                "use_scale": config.scale,
                "dtype": str_dtype_to_jax(config.dtype),
                "param_dtype": str_dtype_to_jax(config.param_dtype),
                "scale_init": self.config.scale_init.instantiate(InitInterface),
            },
        )

        self.norm = nnx.bridge.ToNNX(vmap_module(norm_class, config.axis, config.num_heads)(**norm_kwargs), rngs=rngs)

        if self.config.bias:
            param_dtype = str_dtype_to_jax(config.param_dtype)
            self.bias = nnx.Param(
                self.config.bias_init.instantiate(InitInterface)(rngs.params(), (config.input_dim,), dtype=param_dtype)
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.config.axis is not None:
            if self.config.bias:
                axis_pos = self.config.axis if self.config.axis >= 0 else x.ndim + self.config.axis
                dtype = str_dtype_to_jax(self.config.dtype)
                shape = (
                    *((1,) * (axis_pos)),
                    self.config.num_heads,
                    *((1,) * (x.ndim - axis_pos - 2)),
                    self.config.input_dim // self.config.num_heads,
                )
                bias = self.bias.astype(dtype).reshape(shape)
            else:
                bias = 0.0
            return self.norm(x).astype(x.dtype) + bias
        else:
            if self.config.bias:
                dtype = str_dtype_to_jax(self.config.dtype)
                bias = self.bias.astype(dtype).reshape(
                    self.config.num_heads, self.config.input_dim // self.config.num_heads
                )
            else:
                bias = 0.0
            return (
                self.norm(
                    x.reshape(*x.shape[:-1], self.config.num_heads, self.config.input_dim // self.config.num_heads)
                ).astype(x.dtype)
                + bias
            ).reshape(*x.shape)


class MultiHeadNormLayerRepeated(nnx.Module):
    """deprecated."""

    config: MultiHeadLayerNormConfig

    def __init__(self, config: MultiHeadLayerNormConfig, rngs: nnx.Rngs):
        """Create a multi-head norm layer.

        Effectively vmaps a norm layer over the specified axis.

        Args:
            config: MultiHeadLayerNormConfig for the layer
            rngs:   Random Number Generator Iterator for nnx.
        """
        self.config = config

        if config.norm_type == "layernorm":
            norm_class, norm_kwargs = (
                nnx.LayerNorm,
                {
                    "epsilon": config.eps,
                    "use_bias": config.bias,
                    "use_scale": config.scale,
                    "dtype": str_dtype_to_jax(config.dtype),
                    "param_dtype": str_dtype_to_jax(config.param_dtype),
                    "bias_init": self.config.bias_init.instantiate(InitInterface),
                    "scale_init": self.config.scale_init.instantiate(InitInterface),
                },
            )
        elif config.norm_type == "rmsnorm":
            norm_class, norm_kwargs = (
                nnx.RMSNorm,
                {
                    "epsilon": config.eps,
                    "use_scale": config.scale,
                    "dtype": str_dtype_to_jax(config.dtype),
                    "param_dtype": str_dtype_to_jax(config.param_dtype),
                    "scale_init": self.config.scale_init.instantiate(InitInterface),
                },
            )
        else:
            raise NotImplementedError

        for i in range(self.config.num_heads):
            norm = norm_class(config.input_dim // config.num_heads, rngs=rngs, **norm_kwargs)
            setattr(self, f"norm{i}", norm)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.config.axis is not None:
            return jnp.concatenate(
                [
                    getattr(self, f"norm{i}")(jnp.take(x, i, axis=self.config.axis))
                    for i in range(self.config.num_heads)
                ],
                axis=self.config.axis,
            ).astype(x.dtype)
        else:
            x = x.reshape(*x.shape[:-1], self.config.num_heads, self.config.input_dim // self.config.num_heads)
            return (
                jnp.concatenate([getattr(self, f"norm{i}")(x[..., i, :]) for i in range(self.config.num_heads)])
                .reshape(*x.shape)
                .astype(x.dtype)
            )


@register
class LayerNorm(NormInterface):
    config: LayerNormConfig

    def __init__(self, config: LayerNormConfig, *, rngs: nnx.Rngs):
        """Create a norm layer.

        Args:
            config: LayerNormConfig for the layer
            rngs:   Random Number Generator Iterator for nnx.
        """
        self.config = config
        self.norm = nnx.LayerNorm(
            num_features=config.input_dim,
            epsilon=config.eps,
            dtype=str_dtype_to_jax(config.dtype),
            param_dtype=str_dtype_to_jax(config.param_dtype),
            use_scale=self.config.scale,
            use_bias=self.config.bias,
            bias_init=self.config.bias_init.instantiate(InitInterface),
            scale_init=self.config.scale_init.instantiate(InitInterface),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.norm(x).astype(x.dtype)


@register
class RMSNorm(NormInterface):
    config: RMSNormConfig

    def __init__(self, config: LayerNormConfig, *, rngs: nnx.Rngs):
        """Create a norm layer.

        Args:
            config: RMSNormConfig for the layer
            rngs:   Random Number Generator Iterator for nnx.
        """
        self.config = config
        self.norm = nnx.RMSNorm(
            num_features=config.input_dim,
            epsilon=config.eps,
            dtype=str_dtype_to_jax(config.dtype),
            param_dtype=str_dtype_to_jax(config.param_dtype),
            use_scale=self.config.scale,
            scale_init=self.config.scale_init.instantiate(InitInterface),
            rngs=rngs,
        )
        if self.config.bias:
            param_dtype = str_dtype_to_jax(config.param_dtype)
            self.bias = nnx.Param(
                self.config.bias_init.instantiate(InitInterface)(
                    rngs.params(), (self.config.input_dim,), dtype=param_dtype
                )
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.config.bias:
            dtype = str_dtype_to_jax(self.config.dtype)
            ones_tuple = (1,) * (x.ndim - 1)
            bias = self.bias.astype(dtype).reshape(*ones_tuple, self.config.input_dim)
        else:
            bias = 0.0
        return self.norm(x).astype(x.dtype) + bias


@register
class Identity(NormInterface):
    config: IdentityConfig

    def __init__(self, config: IdentityConfig, *, rngs: nnx.Rngs):
        self.config = config

    def __call__(self, x: jax.Array) -> jax.Array:
        return x
