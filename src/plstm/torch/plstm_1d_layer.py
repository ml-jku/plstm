import torch
from ..config.plstm_1d_layer import pLSTM1DLayerConfig

from .longrange_transition_layer import LongRangeTransitionLayer
from .source_mark_layer import (
    SourceLayer,
    MarkLayer,
    DirectLayer,
)
from .query_key_value import (
    QueryLayer,
    KeyLayer,
    ValueLayer,
)
from .passthrough import PassthroughLayer
from .convolution import Convolution1DLayer
from .plstm_1d import pLSTM1D_fwbw
from .norm import NormInterface
from ..util import log2
from compoconf import register
from .interfaces import ResidualModule
from .initialization import InitInterface


@register
class pLSTM1DLayer(ResidualModule):
    config: pLSTM1DLayerConfig

    def __init__(self, config: pLSTM1DLayerConfig):
        torch.nn.Module.__init__(self)
        self.config = config

        if self.config.additional_convolution:
            self.conv = Convolution1DLayer(self.config.convolution)
        self.source = SourceLayer(self.config.source)
        self.transition = LongRangeTransitionLayer(self.config.transition)
        self.mark = MarkLayer(self.config.mark)
        self.direct = DirectLayer(self.config.direct)
        self.query = QueryLayer(self.config.query)
        if not self.config.tie_query_key:
            self.key = KeyLayer(self.config.key)
        self.value_ = ValueLayer(self.config.value)

        if self.config.mhnorm is not None:
            self.mhnorm = self.config.mhnorm.instantiate(NormInterface)

        if self.config.outprojection:
            self.outprojection = torch.nn.Linear(
                self.config.num_heads * self.config.JO * self.config.DV,
                self.config.input_dim,
                bias=self.config.outprojection_bias,
            )

        if self.config.additional_passthrough:
            self.passthrough = PassthroughLayer(self.config.passthrough)

    def reset_parameters(self, *args, **kwargs):
        if self.config.outprojection:
            self.config.outprojection_weight_init.instantiate(InitInterface)(self.outprojection.weight)
            if self.config.outprojection_bias:
                self.config.outprojection_bias_init.instantiate(InitInterface)(self.outprojection.bias)
        for name, m in self.named_children():
            if name != "outprojection":
                m.reset_parameters(*args, **kwargs)

    def _conv_input(self, x, c, part: str):
        if self.config.additional_convolution and part in self.config.convolution_inputs:
            return c
        else:
            return x

    def forward(self, x, initial_state=None, return_state: bool = False, **kwargs) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.config.input_dim

        if 2 ** log2(x.shape[-2]) != x.shape[-2]:
            padpow2 = 2 ** (log2(x.shape[-2] - 1) + 1) - x.shape[-2]
            T_padded = T + padpow2
            x_pad = torch.cat([x, torch.zeros_like(x)[..., :padpow2, :]], dim=-2)
        else:
            T_padded = T
            x_pad = x
            padpow2 = 0

        if self.config.additional_convolution:
            c_h = self.conv(x_pad).reshape(B, T_padded, D)
        else:
            c_h = None
        x_h = x_pad.reshape(B, T_padded, D)

        if initial_state is not None:
            C0 = initial_state
        else:
            C0 = None
        s = self.source(self._conv_input(x_h, c_h, "S"))  # B, T, H, JT, JK, JV
        t = self.transition(self._conv_input(x_h, c_h, "T"))  # B, T, H, JT, JT
        m = self.mark(self._conv_input(x_h, c_h, "M"))  # B, T, H, JO, JQ, JT
        d = self.direct(self._conv_input(x_h, c_h, "D"))  # B, T, H, JO, JQ, JK, JV
        q = self.query(self._conv_input(x_h, c_h, "Q"))  # B, T, H, DK, JQ
        if self.config.tie_query_key:
            k = q
        else:
            k = self.key(self._conv_input(x_h, c_h, "K"))  # B, T, H, DK, JK
        v = self.value_(self._conv_input(x_h, c_h, "V"))  # B, T, H, DV, JV

        # transpose head and sequence
        s = s.transpose(1, 2).reshape((-1, s.shape[1], *s.shape[3:]))
        t = t.transpose(1, 2).reshape((-1, t.shape[1], *t.shape[3:]))
        m = m.transpose(1, 2).reshape((-1, m.shape[1], *m.shape[3:]))
        d = d.transpose(1, 2).reshape((-1, d.shape[1], *d.shape[3:]))

        q = q.transpose(1, 2).reshape((-1, q.shape[1], *q.shape[3:]))
        k = k.transpose(1, 2).reshape((-1, k.shape[1], *k.shape[3:]))
        v = v.transpose(1, 2).reshape((-1, v.shape[1], *v.shape[3:]))

        o = pLSTM1D_fwbw(
            q,
            k,
            v,
            s,
            t,
            m,
            d,
            C_initial=C0,
            return_last_C=return_state,
            levels=min(self.config.levels, log2(T_padded)),
        )
        if return_state:
            o, C = o
        o = o.reshape(x.shape[0], self.config.num_heads, o.shape[1], -1)

        if self.config.mhnorm:
            out = self.mhnorm(o).transpose(1, 2)
        else:
            out = o.transpose(1, 2)
        out = out.reshape(B, T_padded, D)[:, :T]

        if self.config.outprojection:
            out = self.outprojection(out)
        if self.config.additional_passthrough:
            out += self.passthrough(self._conv_input(x, c_h.view(B, T_padded, D) if c_h is not None else c_h, "P"))
        out = out.view(x.shape)
        if return_state:
            return out, C
        else:
            return out
