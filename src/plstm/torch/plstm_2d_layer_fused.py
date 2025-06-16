import torch

from ..config.plstm_2d_layer_fused import pLSTM2DLayerFusedConfig
from ..util import log2

from .query_key_value import QueryLayer, KeyLayer, ValueLayer
from .source_mark_layer import SourceLayer, MarkLayer, DirectLayer
from .longrange_transition_layer import (
    LongRangeTransitionLayer,
)
from .passthrough import PassthroughLayer
from .convolution import Convolution2DLayer
from .plstm_2d_fused import pLSTM2D_parallel_fused_torch
from .norm import NormInterface
from compoconf import register
from .interfaces import ResidualModule
from .initialization import InitInterface


@register
class pLSTM2DLayerFused(ResidualModule):
    config: pLSTM2DLayerFusedConfig

    def __init__(self, config: pLSTM2DLayerFusedConfig):
        torch.nn.Module.__init__(self)
        self.config = config

        if self.config.additional_convolution:
            self.convolution = Convolution2DLayer(self.config.convolution)

        if self.config.mode == "P":
            self.orientation_bias = torch.nn.Parameter(torch.zeros([4, self.config.num_heads]))
            orientation_weight_shape = [
                4,
                self.config.num_heads,
                self.config.input_dim,
            ]
            self.orientation_weight = torch.nn.Parameter(torch.zeros(orientation_weight_shape))

            self.source = SourceLayer(self.config.source)
            self.transition = LongRangeTransitionLayer(self.config.transition)
            self.mark = MarkLayer(self.config.mark)

        elif self.config.mode == "D":
            self.source_r = SourceLayer(self.config.source)
            self.source_d = SourceLayer(self.config.source)

            self.transition_r = LongRangeTransitionLayer(self.config.transition)
            self.transition_d = LongRangeTransitionLayer(self.config.transition)
            self.transition_rd = LongRangeTransitionLayer(self.config.transition)

            self.mark_l = MarkLayer(self.config.mark)
            self.mark_u = MarkLayer(self.config.mark)

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
        self.reset_parameters()

    def reset_parameters(self, *args, **kwargs):
        with torch.no_grad():
            if self.config.mode == "P":
                self.config.pmode_orientation_bias_init.instantiate(InitInterface)(self.orientation_bias)
                self.config.pmode_orientation_weight_init.instantiate(InitInterface)(self.orientation_weight)

        for _, m in self.named_children():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters(*args, **kwargs)

    @staticmethod
    def _transpose_hxy(*args):
        ret = []
        for arg in args:
            if arg is None:
                ret.append(None)
            else:
                ret.append(
                    arg.permute(0, 3, 1, 2, *range(4, arg.ndim)).reshape(-1, arg.shape[1], arg.shape[2], *arg.shape[4:])
                )
        return ret

    def _conv_input(self, x, c, part: str):
        if self.config.additional_convolution and part in self.config.convolution_inputs:
            return c
        else:
            return x

    def _gating_function(self, x):
        if self.config.gating_function == "silu":
            return torch.nn.functional.silu(x)
        else:
            raise ValueError("Bad gating function")

    def forward(
        self,
        x,
        **kwargs,
    ) -> torch.Tensor:
        B, X, Y, D = x.shape
        H = self.config.num_heads

        if self.config.additional_convolution:
            c = self.convolution(x)
        else:
            c = None

        q = self.query(self._conv_input(x, c, "Q"))  # B, X, Y, H, DK, JQ
        if self.config.tie_query_key:
            k = q
        else:
            k = self.key(self._conv_input(x, c, "K"))  # B, X, Y, H, DK, JK
        v = self.value_(self._conv_input(x, c, "V"))  # B, X, Y, H, DV, JV

        d = self.direct(self._conv_input(x, c, "D"))

        if self.config.mode == "P":
            orientation = torch.sigmoid(
                self.orientation_bias
                + self.config.pmode_orientation_scale
                * torch.einsum("ohx,...x->...oh", self.orientation_weight, self._conv_input(x, c, "O"))
                / self.config.input_dim
            ).reshape(*x.shape[:-1], -1)
            s_r = orientation[..., None, None, None] * self.source(self._conv_input(x, c, "S"))
            s_d = 1.0 - s_r
            t_rl = orientation[..., None, None] * self.transition(self._conv_input(x, c, "T"))
            t_ru = t_rl
            t_dl = 1 - t_rl
            t_du = 1 - t_rl
            m_l = self.mark(self._conv_input(x, c, "M"))
            m_u = m_l
        else:
            s_r = self.source_r(self._conv_input(x, c, "S"))
            s_d = self.source_d(self._conv_input(x, c, "S"))
            t_rl = self.transition_r(self._conv_input(x, c, "T"))
            t_du = self.transition_d(self._conv_input(x, c, "T"))
            t_rd = self.transition_rd(self._conv_input(x, c, "T"))
            t_ru = None if self.config.dmode_chirality == "R" else t_rd
            t_dl = None if self.config.dmode_chirality == "L" else t_rd
            m_l = self.mark_l(self._conv_input(x, c, "M"))
            m_u = self.mark_u(self._conv_input(x, c, "M"))

        levels = min(self.config.levels, 1 + log2(X - 1), 1 + log2(Y - 1))

        if X % (1 << levels) != 0 or Y % (1 << levels) != 0:
            levels = min(self.config.levels, 1 + log2(X - 1), 1 + log2(Y - 1))
            X_padded = X + ((1 << levels) - X % (1 << levels))
            Y_padded = Y + ((1 << levels) - Y % (1 << levels))

            def _pad(ar):
                return (
                    torch.nn.functional.pad(ar, (0, 0) * (ar.ndim - 3) + (0, Y_padded - Y, 0, X_padded - X))
                    if ar is not None
                    else None
                )

            q = _pad(q)
            k = _pad(k)
            v = _pad(v)
            s_r = _pad(s_r)
            s_d = _pad(s_d)
            t_rl = _pad(t_rl)
            t_du = _pad(t_du)
            t_dl = _pad(t_dl)
            t_ru = _pad(t_ru)
            m_l = _pad(m_l)
            m_u = _pad(m_u)
            d = _pad(d)

        else:
            X_padded = X
            Y_padded = Y

        # transpose heads first
        q, k, v, s_r, s_d, t_rl, t_du, t_dl, t_ru, m_l, m_u, d = self._transpose_hxy(
            q, k, v, s_r, s_d, t_rl, t_du, t_dl, t_ru, m_l, m_u, d
        )

        # Reshape inputs to match the expected shapes for pLSTM2D_fwbw
        q_reshaped = q.reshape(B * H, X_padded, Y_padded, self.config.DK, self.config.JQ)
        k_reshaped = k.reshape(B * H, X_padded, Y_padded, self.config.DK, self.config.JK)
        v_reshaped = v.reshape(B * H, X_padded, Y_padded, self.config.DV, self.config.JV)
        s_r_reshaped = s_r.reshape(B * H, 4, X_padded, Y_padded, self.config.JT, self.config.JK, self.config.JV)
        s_d_reshaped = s_d.reshape(B * H, 4, X_padded, Y_padded, self.config.JT, self.config.JK, self.config.JV)
        t_rl_reshaped = t_rl.reshape(B * H, 4, X_padded, Y_padded, self.config.JT, self.config.JT)
        t_du_reshaped = t_du.reshape(B * H, 4, X_padded, Y_padded, self.config.JT, self.config.JT)
        t_dl_reshaped = (
            t_dl.reshape(B * H, 4, X_padded, Y_padded, self.config.JT, self.config.JT) if t_dl is not None else None
        )
        t_ru_reshaped = (
            t_ru.reshape(B * H, 4, X_padded, Y_padded, self.config.JT, self.config.JT) if t_ru is not None else None
        )
        m_l_reshaped = m_l.reshape(B * H, 4, X_padded, Y_padded, self.config.JO, self.config.JQ, self.config.JT)
        m_u_reshaped = m_u.reshape(B * H, 4, X_padded, Y_padded, self.config.JO, self.config.JQ, self.config.JT)
        d_reshaped = d.reshape(
            B * H, X_padded, Y_padded, self.config.JO, self.config.JQ, self.config.JK, self.config.JV
        )

        out = pLSTM2D_parallel_fused_torch(
            q_reshaped,
            k_reshaped,
            v_reshaped,
            s_r_reshaped,
            s_d_reshaped,
            t_rl_reshaped,
            t_du_reshaped,
            t_dl_reshaped,
            t_ru_reshaped,
            m_l_reshaped,
            m_u_reshaped,
            d_reshaped,
            levels=levels,
        )

        out = out.reshape(B, H, X_padded, Y_padded, self.config.DV * self.config.JO)
        out = out[:, :, :X, :Y]

        if self.config.mhnorm is not None:
            out = self.mhnorm(out)

        out = out.reshape(B, H, X, Y, -1).permute(0, 2, 3, 1, 4).reshape(B, X, Y, -1)

        if self.config.outprojection:
            out = self.outprojection(out)

        if self.config.additional_passthrough:
            out = out + self.passthrough(x)

        return out
