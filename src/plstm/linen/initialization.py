from typing import Any
from compoconf import register, register_interface, RegistrableConfigInterface
from jax.random import PRNGKey
from flax import linen as nn
from abc import abstractmethod
import jax.numpy as jnp
from operator import itemgetter

from ..util import prod, positive_index, revert_map
from ..config.initialization import (
    ConstantInitConfig,
    OnesInitConfig,
    ZerosInitConfig,
    DiagonalInitConfig,
    LinspaceInitConfig,
    NormalInitConfig,
    TruncatedNormalInitConfig,
    WangInitConfig,
    SmallInitConfig,
)


@register_interface
class InitInterface(RegistrableConfigInterface):
    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def __call__(self, key: PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        raise NotImplementedError


@register
class ZerosInit(InitInterface):
    config: ZerosInitConfig

    def __init__(self, config: ZerosInitConfig):
        self.config = config
        self._initializer = nn.initializers.zeros

    def __call__(self, key: PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        return self._initializer(key, shape, dtype=dtype)


@register
class OnesInit(InitInterface):
    config: OnesInitConfig

    def __init__(self, config: OnesInitConfig):
        self.config = config
        self._initializer = nn.initializers.ones

    def __call__(self, key: PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        return self._initializer(key, shape, dtype=dtype)


@register
class ConstantInit(InitInterface):
    config: ConstantInitConfig

    def __init__(self, config: ConstantInitConfig):
        self.config = config
        self._initializer = nn.initializers.constant(self.config.value)

    def __call__(self, key: PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        return self._initializer(key, shape, dtype=dtype)


@register
class DiagonalInit(InitInterface):
    config: DiagonalInitConfig

    def __init__(self, config: DiagonalInitConfig):
        self.config = config

    def __call__(self, key: PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        ia = (self.config.in_axes,) if isinstance(self.config.in_axes, int) else self.config.in_axes
        oa = (self.config.out_axes,) if isinstance(self.config.out_axes, int) else self.config.out_axes
        ia, oa = (tuple(map(lambda x: positive_index(x, len(shape)), ax)) for ax in (ia, oa))
        ba = tuple(ax for ax in range(len(shape)) if (ax not in ia and ax not in oa))

        in_size = prod(*(shape[ax] for ax in ia))
        out_size = prod(*(shape[ax] for ax in oa))
        batch_size = prod(*shape) // in_size // out_size

        assert max(in_size, out_size) % min(in_size, out_size) == 0, "Need divisible sizes for eye init of source"

        diag = jnp.tile(jnp.eye(min(in_size, out_size)), (max(1, in_size // out_size), max(1, out_size // in_size)))
        res = diag.reshape(1, in_size, out_size).repeat(batch_size, axis=0)
        gen_shape = tuple(shape[ax] for ax in ba) + tuple(shape[ax] for ax in ia) + tuple(shape[ax] for ax in oa)
        return res.reshape(gen_shape).transpose(
            *map(itemgetter(1), sorted(revert_map(tuple((*ba, *ia, *oa))), key=itemgetter(0)))
        )


@register
class LinspaceInit(InitInterface):
    config: LinspaceInitConfig

    def __init__(self, config: LinspaceInitConfig):
        self.config = config

    def __call__(self, key: PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        axis = self.config.axis if self.config.axis >= 0 else len(shape) + self.config.axis
        size = shape[axis]
        assert size > 1, "For Linspace init, need at minimum size to to cover low and high limit."
        step = (self.config.high - self.config.low) / (size - 1)

        return jnp.tile(
            jnp.arange(self.config.low, self.config.high + step, step).reshape(
                *(1 for _ in shape[:axis]), size, *(1 for _ in shape[axis + 1 :])
            ),
            (*shape[:axis], 1, *shape[axis + 1 :]),
        )


@register
class NormalInit(InitInterface):
    config: NormalInitConfig

    def __init__(self, config: NormalInitConfig):
        self.config = config
        self._initializer = nn.initializers.normal(stddev=self.config.stddev)

    def __call__(self, key: PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        return self.config.mean + self._initializer(key, shape, dtype=dtype)


@register
class TruncatedNormalInit(InitInterface):
    config: TruncatedNormalInitConfig

    def __init__(self, config: TruncatedNormalInitConfig):
        self.config = config
        self._initializer = nn.initializers.truncated_normal(
            stddev=self.config.stddev, lower=self.config.lower, upper=self.config.upper
        )

    def __call__(self, key: PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        return self.config.mean + self._initializer(key, shape, dtype=dtype)


@register
class WangInit(InitInterface):
    config: WangInitConfig

    def __init__(self, config: WangInitConfig):
        self.config = config
        scale = 2 * self.config.mup_init_scale / config.num_blocks
        self._initializer = nn.initializers.variance_scaling(
            scale=scale**2,
            mode="fan_in",
            distribution="normal",
            in_axis=self.config.axis,
        )

    def __call__(self, key: PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        return self._initializer(key, shape, dtype=dtype)


@register
class SmallInit(InitInterface):
    config: SmallInitConfig

    def __init__(self, config: SmallInitConfig):
        self.config = config
        scale = self.config.mup_init_scale * (2 / 5) ** (1 / 2)
        self._initializer = nn.initializers.variance_scaling(
            scale=scale**2,
            mode="fan_in",
            distribution="normal",
            in_axis=self.config.axis,
        )

    def __call__(self, key: PRNGKey, shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
        return self._initializer(key, shape, dtype=dtype)
