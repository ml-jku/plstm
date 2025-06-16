from typing import Any
from compoconf import register, register_interface, RegistrableConfigInterface
import torch
import torch.nn.init as init
from abc import abstractmethod
from operator import itemgetter

from ..util import positive_index, revert_map, prod
from ..config.initialization import (
    ConstantInitConfig,
    OnesInitConfig,
    ZerosInitConfig,
    LinspaceInitConfig,
    NormalInitConfig,
    TruncatedNormalInitConfig,
    WangInitConfig,
    SmallInitConfig,
    DiagonalInitConfig,
)


@register_interface
class InitInterface(RegistrableConfigInterface):
    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> None:
        """Apply initialization in-place to the provided tensor."""
        raise NotImplementedError


@register
class ZerosInit(InitInterface):
    config: ZerosInitConfig

    def __init__(self, config: ZerosInitConfig):
        self.config = config

    def __call__(self, tensor: torch.Tensor) -> None:
        """Apply zeros initialization in-place."""
        init.zeros_(tensor)


@register
class OnesInit(InitInterface):
    config: OnesInitConfig

    def __init__(self, config: OnesInitConfig):
        self.config = config

    def __call__(self, tensor: torch.Tensor) -> None:
        """Apply ones initialization in-place."""
        init.ones_(tensor)


@register
class ConstantInit(InitInterface):
    config: ConstantInitConfig

    def __init__(self, config: ConstantInitConfig):
        self.config = config

    def __call__(self, tensor: torch.Tensor) -> None:
        """Apply constant initialization in-place."""
        init.constant_(tensor, self.config.value)


@register
class LinspaceInit(InitInterface):
    config: LinspaceInitConfig

    def __init__(self, config: LinspaceInitConfig):
        self.config = config

    def __call__(self, tensor: torch.Tensor) -> None:
        """Apply linspace initialization in-place."""
        axis = self.config.axis
        if axis < 0:
            axis = len(tensor.shape) + axis

        size = tensor.shape[axis]
        assert size > 1, "For Linspace init, need at minimum size to cover lower and upper limit."

        # Create linspace values
        values = torch.linspace(self.config.low, self.config.high, size, device=tensor.device)

        # Create shape for broadcasting
        shape = [1] * len(tensor.shape)
        shape[axis] = size

        # Reshape values for broadcasting
        values = values.view(shape)

        # Broadcast and assign
        tensor.copy_(values.expand_as(tensor))


@register
class NormalInit(InitInterface):
    config: NormalInitConfig

    def __init__(self, config: NormalInitConfig):
        self.config = config

    def __call__(self, tensor: torch.Tensor) -> None:
        """Apply normal initialization in-place."""
        init.normal_(tensor, mean=self.config.mean, std=self.config.stddev)


@register
class TruncatedNormalInit(InitInterface):
    config: TruncatedNormalInitConfig

    def __init__(self, config: TruncatedNormalInitConfig):
        self.config = config

    def __call__(self, tensor: torch.Tensor) -> None:
        """Apply truncated normal initialization in-place.

        PyTorch doesn't have a built-in truncated normal initializer, so
        we implement it manually.
        """
        mean = self.config.mean
        std = self.config.stddev
        lower = self.config.lower
        upper = self.config.upper

        # Generate normal distribution
        init.normal_(tensor, mean=mean, std=std)

        # Truncate values outside the range
        with torch.no_grad():
            tensor.clamp_(min=mean + lower * std, max=mean + upper * std)


@register
class DiagonalInit(InitInterface):
    config: DiagonalInitConfig

    def __init__(self, config: DiagonalInitConfig):
        self.config = config

    def __call__(self, x: torch.Tensor) -> None:
        shape = x.shape
        ia = (self.config.in_axes,) if isinstance(self.config.in_axes, int) else self.config.in_axes
        oa = (self.config.out_axes,) if isinstance(self.config.out_axes, int) else self.config.out_axes
        ia, oa = (tuple(map(lambda x: positive_index(x, len(shape)), ax)) for ax in (ia, oa))
        ba = tuple(ax for ax in range(len(shape)) if (ax not in ia and ax not in oa))

        in_size = prod(*(shape[ax] for ax in ia))
        out_size = prod(*(shape[ax] for ax in oa))
        batch_size = prod(*shape) // in_size // out_size

        assert max(in_size, out_size) % min(in_size, out_size) == 0, "Need divisible sizes for eye init of source"

        diag = torch.tile(torch.eye(min(in_size, out_size)), (max(1, in_size // out_size), max(1, out_size // in_size)))
        res = diag.reshape(1, in_size, out_size).tile((batch_size, 1, 1))
        gen_shape = tuple(shape[ax] for ax in ba) + tuple(shape[ax] for ax in ia) + tuple(shape[ax] for ax in oa)
        x.copy_(
            res.reshape(gen_shape).permute(
                *map(itemgetter(1), sorted(revert_map(tuple((*ba, *ia, *oa))), key=itemgetter(0)))
            ),
        )


@register
class WangInit(InitInterface):
    config: WangInitConfig

    def __init__(self, config: WangInitConfig):
        self.config = config

    def __call__(self, tensor: torch.Tensor) -> None:
        """Apply Wang initialization in-place.

        Based on EleutherAI's GPT-NeoX implementation.
        """
        # This is a placeholder implementation
        # The actual Wang initialization would need to be implemented
        # based on the GPT-NeoX implementation
        axis = self.config.axis
        if axis < 0:
            axis = len(tensor.shape) + axis

        fan_in = tensor.shape[axis]
        scale = 2 * self.config.mup_init_scale / (fan_in) ** (1 / 2) / self.config.num_blocks
        init.normal_(tensor, mean=0.0, std=scale)


@register
class SmallInit(InitInterface):
    config: SmallInitConfig

    def __init__(self, config: SmallInitConfig):
        self.config = config

    def __call__(self, tensor: torch.Tensor) -> None:
        """Apply small initialization in-place."""
        axis = self.config.axis
        if axis < 0:
            axis = len(tensor.shape) + axis

        fan_in = tensor.shape[axis]
        scale = self.config.mup_init_scale / (5 / 2 * fan_in) ** (1 / 2)
        init.normal_(tensor, mean=0.0, std=scale)
