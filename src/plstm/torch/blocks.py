import torch
from ..config.blocks import PreUpProjectionBlockConfig, PostUpProjectionBlockConfig

from .norm import NormInterface
from compoconf import register
from .interfaces import ResidualModule
from .initialization import InitInterface
from .dtype import str_dtype_to_torch


@register
class PreUpProjectionBlock(ResidualModule):
    config: PreUpProjectionBlockConfig

    def __init__(self, config: PreUpProjectionBlockConfig):
        ResidualModule.__init__(self, config)
        self.config = config

        self.norm = config.norm.instantiate(NormInterface)

        self.upproj = torch.nn.Linear(
            config.input_dim,
            2 * config.inner_input_dim if config.gated else config.inner_input_dim,
            dtype=str_dtype_to_torch(config.param_dtype),
        )

        self.interaction_module = config.interaction_module.instantiate(ResidualModule)

        self.downproj = torch.nn.Linear(
            config.inner_input_dim, config.input_dim, dtype=str_dtype_to_torch(config.param_dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.config.upproj_weight_init.instantiate(InitInterface)(self.upproj.weight)
            self.config.upproj_bias_init.instantiate(InitInterface)(self.upproj.bias)
            self.config.downproj_weight_init.instantiate(InitInterface)(self.downproj.weight)
            self.config.downproj_bias_init.instantiate(InitInterface)(self.downproj.bias)

            self.interaction_module.reset_parameters()
            self.norm.reset_parameters()

    def _gating_function(self, x):
        if self.config.gating_function == "silu":
            return torch.nn.functional.silu(x)
        elif self.config.gating_function == "gelu":
            return torch.nn.functionl.gelu(x)
        else:
            raise ValueError("Bad gating function")

    def forward(self, x, **kwargs):
        with torch.autocast(device_type=x.device.type, dtype=str_dtype_to_torch(self.config.dtype)):
            x1 = self.norm(x)
            x2 = self.upproj(x1)

            if self.config.gated:
                x3, x4 = (
                    x2[..., : self.config.inner_input_dim],
                    x2[..., self.config.inner_input_dim :],
                )
            else:
                x3 = x2

            y = self.interaction_module(x3, **kwargs)

            if self.config.gated:
                y2 = y.reshape(*x4.shape) * self._gating_function(x4)
            else:
                y2 = y.reshape(*x3.shape)

            y3 = self.downproj(y2)

            if self.config.skip:
                y4 = y3 + x
            else:
                y4 = y3

        return y4


@register
class PostUpProjectionBlock(ResidualModule):
    config: PostUpProjectionBlockConfig

    def __init__(self, config: PostUpProjectionBlockConfig):
        ResidualModule.__init__(self, config)
        self.config = config

        self.norm = config.norm.instantiate(NormInterface)
        self.norm2 = config.norm.instantiate(NormInterface)

        self.drop_path = torch.nn.Dropout(self.config.drop_path_rate)

        if self.config.gated:
            self.upproj = torch.nn.Linear(
                self.config.input_dim,
                2 * self.config.inner_input_dim,
                dtype=str_dtype_to_torch(config.param_dtype),
            )
        else:
            self.upproj = torch.nn.Linear(
                self.config.input_dim,
                self.config.inner_input_dim,
                dtype=str_dtype_to_torch(config.param_dtype),
            )

        self.interaction_module = self.config.interaction_module.instantiate(ResidualModule)

        self.downproj = torch.nn.Linear(
            self.config.inner_input_dim,
            self.config.input_dim,
            dtype=str_dtype_to_torch(config.param_dtype),
        )

        if self.config.use_scale:
            self.scale1 = self.config.scale.instantiate(ResidualModule)
            self.scale2 = self.config.scale.instantiate(ResidualModule)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.config.upproj_weight_init.instantiate(InitInterface)(self.upproj.weight)
            self.config.upproj_bias_init.instantiate(InitInterface)(self.upproj.bias)
            self.config.downproj_weight_init.instantiate(InitInterface)(self.downproj.weight)
            self.config.downproj_bias_init.instantiate(InitInterface)(self.downproj.bias)

            self.interaction_module.reset_parameters()
            self.norm.reset_parameters()
            self.norm2.reset_parameters()

            if self.config.use_scale:
                self.scale1.reset_parameters()
                self.scale2.reset_parameters()

    def _gating_function(self, x):
        if self.config.gating_function == "silu":
            return torch.nn.functional.silu(x)
        elif self.config.gating_function == "gelu":
            return torch.nn.functional.gelu(x)
        else:
            raise ValueError("Bad gating function")

    def _drop_path(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return x
        else:
            return self.drop_path(torch.ones((x.shape[0], *([1] * (x.ndim - 1))), device=x.device)) * x

    def forward(self, x, deterministic: bool = False, **kwargs):
        with torch.autocast(device_type=x.device.type, dtype=str_dtype_to_torch(self.config.dtype)):
            x1 = self.norm(x)
            y1 = self.interaction_module(x1, **kwargs)

            if self.config.skip:
                if self.config.use_scale:
                    y1 = self.scale1(y1)
                x2 = self._drop_path(y1, deterministic=deterministic) + x
            else:
                x2 = y1

            x3 = self.norm2(x2)
            x4 = self.upproj(x3)

            if self.config.gated:
                x5, x6 = (
                    x4[..., : self.config.inner_input_dim],
                    x4[..., self.config.inner_input_dim :],
                )
            else:
                x5 = x4

            if self.config.gated:
                y2 = x5 * self._gating_function(x6)
            else:
                y2 = self._gating_function(x5)

            y3 = self.downproj(y2)

            if self.config.skip:
                if self.config.use_scale:
                    y3 = self.scale2(y3)
                y4 = self._drop_path(y3, deterministic=deterministic) + x2
            else:
                y4 = y3

        return y4
