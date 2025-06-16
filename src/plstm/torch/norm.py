import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from compoconf import RegistrableConfigInterface, register, register_interface

from ..config.norm import (
    MultiHeadLayerNormConfig,
    LayerNormConfig,
    MultiHeadRMSNormConfig,
    RMSNormConfig,
    IdentityConfig,
)
from .dtype import str_dtype_to_torch
from .initialization import InitInterface


@register_interface
class NormInterface(nn.Module, RegistrableConfigInterface):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@register
class MultiHeadLayerNorm(NormInterface):
    config: MultiHeadLayerNormConfig

    def __init__(self, config: MultiHeadLayerNormConfig):
        """Create a multi-head layer norm.

        Effectively applies layer normalization over the specified axis for each head.

        Args:
            config: MultiHeadLayerNormConfig for the layer
        """
        nn.Module.__init__(self)
        self.config = config

        # Initialize parameters if needed
        if config.scale:
            self.scale = nn.Parameter(
                torch.ones(
                    config.num_heads, config.input_dim // config.num_heads, dtype=str_dtype_to_torch(config.param_dtype)
                )
            )
        else:
            self.register_parameter("scale", None)

        if config.bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    config.num_heads, config.input_dim // config.num_heads, dtype=str_dtype_to_torch(config.param_dtype)
                )
            )
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        with torch.no_grad():
            if self.config.scale:
                self.config.scale_init.instantiate(InitInterface)(self.scale)
            if self.config.bias:
                self.config.bias_init.instantiate(InitInterface)(self.bias)

    def _norm_fn(self, x: torch.Tensor, scale: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.config.input_dim // self.config.num_heads,),
            weight=scale,
            bias=bias,
            eps=self.config.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.axis is None:
            axis = -2
            xr = x.reshape(*x.shape[:-1], self.config.num_heads, x.shape[-1] // self.config.num_heads)
        else:
            xr = x
            axis = self.config.axis

        # Create vmap function for the normalization
        vmapped_norm = torch.vmap(
            self._norm_fn,
            in_dims=(axis, 0 if self.scale is not None else None, 0 if self.bias is not None else None),
            out_dims=(axis,),
        )

        # Apply normalization
        y = vmapped_norm(xr, self.scale, self.bias)

        if self.config.axis is None:
            y = y.reshape(*x.shape)

        return y


@register
class MultiHeadRMSNorm(NormInterface):
    config: MultiHeadRMSNormConfig

    def __init__(self, config: MultiHeadRMSNormConfig):
        """Create a multi-head RMS norm.

        Effectively applies RMS normalization over the specified axis for each head.

        Args:
            config: MultiHeadRMSNormConfig for the layer
        """
        nn.Module.__init__(self)
        self.config = config

        # Initialize parameters if needed
        if config.scale:
            self.scale = nn.Parameter(
                torch.ones(
                    config.num_heads, config.input_dim // config.num_heads, dtype=str_dtype_to_torch(config.param_dtype)
                )
            )
        else:
            self.register_parameter("scale", None)

        if config.bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    config.num_heads, config.input_dim // config.num_heads, dtype=str_dtype_to_torch(config.param_dtype)
                )
            )
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        with torch.no_grad():
            if self.config.scale:
                self.config.scale_init.instantiate(InitInterface)(self.scale)
            if self.config.bias:
                self.config.bias_init.instantiate(InitInterface)(self.bias)

    def _norm_fn(self, x: torch.Tensor, scale: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor:
        return F.rms_norm(
            x,
            normalized_shape=(self.config.input_dim // self.config.num_heads,),
            weight=scale,
            eps=self.config.eps,
        ) + (0.0 if bias is None else bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.axis is None:
            axis = -2
            xr = x.reshape(*x.shape[:-1], self.config.num_heads, x.shape[-1] // self.config.num_heads)
        else:
            xr = x
            axis = self.config.axis

        # Create vmap function for the normalization
        vmapped_norm = torch.vmap(
            self._norm_fn,
            in_dims=(axis, 0 if self.scale is not None else None, 0 if self.bias is not None else None),
            out_dims=(axis,),
        )

        # Apply normalization
        y = vmapped_norm(xr, self.scale, self.bias)

        if self.config.axis is None:
            y = y.reshape(*x.shape)

        return y


@register
class LayerNorm(NormInterface):
    config: LayerNormConfig

    def __init__(self, config: LayerNormConfig):
        """Create a layer norm.

        Args:
            config: LayerNormConfig for the layer
        """
        nn.Module.__init__(self)
        self.config = config

        # Initialize parameters if needed
        if config.scale:
            self.scale = nn.Parameter(torch.ones(config.input_dim, dtype=str_dtype_to_torch(config.param_dtype)))
        else:
            self.register_parameter("scale", None)

        if config.bias:
            self.bias = nn.Parameter(torch.zeros(config.input_dim, dtype=str_dtype_to_torch(config.param_dtype)))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        with torch.no_grad():
            if self.config.scale:
                self.config.scale_init.instantiate(InitInterface)(self.scale)
            if self.config.bias:
                self.config.bias_init.instantiate(InitInterface)(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use functional layer norm for parallelization
        x = F.layer_norm(
            x,
            normalized_shape=(self.config.input_dim,),
            weight=self.scale if self.config.scale else None,
            bias=self.bias if self.config.bias else None,
            eps=self.config.eps,
        )
        return x.to(dtype=str_dtype_to_torch(self.config.dtype))


@register
class RMSNorm(NormInterface):
    config: RMSNormConfig

    def __init__(self, config: RMSNormConfig):
        """Create an RMS norm layer.

        Args:
            config: RMSNormConfig for the layer
        """
        nn.Module.__init__(self)
        self.config = config

        # Initialize parameters if needed
        if config.scale:
            self.scale = nn.Parameter(torch.ones(config.input_dim, dtype=str_dtype_to_torch(config.param_dtype)))
        else:
            self.register_parameter("scale", None)

        if config.bias:
            self.bias = nn.Parameter(torch.zeros(config.input_dim, dtype=str_dtype_to_torch(config.param_dtype)))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        with torch.no_grad():
            if self.config.scale:
                self.config.scale_init.instantiate(InitInterface)(self.scale)
            if self.config.bias:
                self.config.bias_init.instantiate(InitInterface)(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use functional RMS norm for parallelization
        x = F.rms_norm(
            x,
            normalized_shape=(self.config.input_dim,),
            weight=self.scale if self.config.scale else None,
            eps=self.config.eps,
        ) + (0.0 if self.bias is None else self.bias)
        return x.to(dtype=str_dtype_to_torch(self.config.dtype))


@register
class Identity(NormInterface):
    config: IdentityConfig

    def __init__(self, config: IdentityConfig):
        """Create an identity layer (no normalization).

        Args:
            config: IdentityConfig for the layer
        """
        nn.Module.__init__(self)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Just pass through
        return x
