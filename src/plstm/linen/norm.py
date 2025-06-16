from flax import linen as nn
import jax
import jax.numpy as jnp
from .dtype import str_dtype_to_jax
from compoconf import register_interface, RegistrableConfigInterface, register
from abc import abstractmethod
from .initialization import InitInterface

from ..config.norm import (
    MultiHeadLayerNormConfig,
    LayerNormConfig,
    MultiHeadRMSNormConfig,
    RMSNormConfig,
    IdentityConfig,
)


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
class NormInterface(nn.Module, RegistrableConfigInterface):
    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError


@register
class MultiHeadLayerNorm(NormInterface):
    config: MultiHeadLayerNormConfig

    def setup(self):
        """Create a multi-head norm layer.

        Effectively vmaps a norm layer over the specified axis.
        """
        norm_class, norm_kwargs = (
            nn.LayerNorm,
            {
                "epsilon": self.config.eps,
                "use_bias": self.config.bias,
                "use_scale": self.config.scale,
                "dtype": str_dtype_to_jax(self.config.dtype),
            },
        )

        self.norm = vmap_module(norm_class, self.config.axis, self.config.num_heads)(**norm_kwargs)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.config.axis is not None:
            return self.norm(x).astype(x.dtype)
        else:
            return (
                self.norm(x.reshape(*x.shape[:-1], self.config.num_heads, x.shape[-1] // self.config.num_heads))
                .reshape(*x.shape)
                .astype(x.dtype)
            )


@register
class MultiHeadRMSNorm(NormInterface):
    config: MultiHeadRMSNormConfig

    def setup(self):
        """Create a multi-head norm layer.

        Effectively vmaps a norm layer over the specified axis.
        """
        norm_class, norm_kwargs = (
            nn.RMSNorm,
            {
                "epsilon": self.config.eps,
                "use_scale": self.config.scale,
                "dtype": str_dtype_to_jax(self.config.dtype),
                "param_dtype": str_dtype_to_jax(self.config.param_dtype),
            },
        )

        self.norm = vmap_module(norm_class, self.config.axis, self.config.num_heads)(**norm_kwargs)
        if self.config.bias:
            param_dtype = str_dtype_to_jax(self.config.param_dtype)
            self.bias = self.param(
                "bias",
                self.config.bias_init.instantiate(InitInterface),
                (self.config.input_dim,),
                param_dtype,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.config.axis is not None:
            axis_pos = self.config.axis if self.config.axis >= 0 else x.ndim + self.config.axis
            if self.config.bias:
                dtype = str_dtype_to_jax(self.config.dtype)
                shape = (
                    *((1,) * axis_pos),
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


class MultiHeadNormLayerRepeated(nn.Module):
    """deprecated."""

    config: MultiHeadLayerNormConfig

    def setup(self):
        """Create a multi-head norm layer.

        Effectively vmaps a norm layer over the specified axis.
        """
        self.norms = []

        if self.config.norm_type == "layernorm":
            norm_class, norm_kwargs = (
                nn.LayerNorm,
                {
                    "epsilon": self.config.eps,
                    "use_bias": self.config.bias,
                    "use_scale": self.config.scale,
                    "dtype": str_dtype_to_jax(self.config.dtype),
                },
            )
        elif self.config.norm_type == "rmsnorm":
            norm_class, norm_kwargs = (
                nn.RMSNorm,
                {
                    "epsilon": self.config.eps,
                    "use_scale": self.config.scale,
                    "dtype": str_dtype_to_jax(self.config.dtype),
                    "param_dtype": str_dtype_to_jax(self.config.param_dtype),
                },
            )
        else:
            raise NotImplementedError

        for i in range(self.config.num_heads):
            norm = norm_class(self.config.input_dim // self.config.num_heads, **norm_kwargs)
            self.norms.append(norm)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.config.axis is not None:
            return jnp.concatenate(
                [self.norms[i](jnp.take(x, i, axis=self.config.axis)) for i in range(self.config.num_heads)],
                axis=self.config.axis,
            ).astype(x.dtype)
        else:
            x = x.reshape(*x.shape[:-1], self.config.num_heads, x.shape[-1] // self.config.num_heads)
            return (
                jnp.concatenate([self.norms[i](x[..., i, :]) for i in range(self.config.num_heads)])
                .reshape(*x.shape)
                .astype(x.dtype)
            )


@register
class LayerNorm(NormInterface):
    config: LayerNormConfig

    def setup(self):
        """Create a norm layer."""
        self.norm = nn.LayerNorm(
            epsilon=self.config.eps,
            dtype=str_dtype_to_jax(self.config.dtype),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
            use_scale=self.config.scale,
            use_bias=self.config.bias,
            scale_init=self.config.scale_init.instantiate(InitInterface),
            bias_init=self.config.bias_init.instantiate(InitInterface),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.norm(x).astype(x.dtype)


@register
class RMSNorm(NormInterface):
    config: RMSNormConfig

    def setup(self):
        """Create a norm layer."""
        self.norm = nn.RMSNorm(
            epsilon=self.config.eps,
            dtype=str_dtype_to_jax(self.config.dtype),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
            use_scale=self.config.scale,
            scale_init=self.config.scale_init.instantiate(InitInterface),
        )
        if self.config.bias:
            param_dtype = str_dtype_to_jax(self.config.param_dtype)
            self.bias = self.param(
                "bias", self.config.bias_init.instantiate(InitInterface), (self.config.input_dim,), param_dtype
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

    def setup(self):
        pass

    def __call__(self, x: jax.Array) -> jax.Array:
        return x
