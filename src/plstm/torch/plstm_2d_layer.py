import torch

from ..config.plstm_2d_layer import pLSTM2DLayerConfig
from ..util import log2

from .query_key_value import QueryLayer, KeyLayer, ValueLayer
from .source_mark_layer import SourceLayer, MarkLayer, DirectLayer
from .longrange_transition_layer import (
    LongRangeTransitionLayer,
)
from .passthrough import PassthroughLayer
from .convolution import Convolution2DLayer
from .plstm_2d import pLSTM2D_fwbw
from .norm import NormInterface
from compoconf import register
from .interfaces import ResidualModule
from .initialization import InitInterface


@register
class pLSTM2DLayer(ResidualModule):
    config: pLSTM2DLayerConfig

    def __init__(self, config: pLSTM2DLayerConfig):
        torch.nn.Module.__init__(self)
        self.config = config

        if self.config.additional_convolution:
            self.convolution = Convolution2DLayer(self.config.convolution)

        if self.config.mode == "P":
            self.orientation_bias = torch.nn.Parameter(
                torch.zeros([len(self.config.orientation_combinations), self.config.num_heads])
            )
            self.orientation_weight = torch.nn.Parameter(
                torch.zeros(
                    [
                        len(self.config.orientation_combinations),
                        self.config.num_heads,
                        self.config.input_dim,
                    ]
                )
            )

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

        if self.config.mhnorm:
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
        with torch.no_grad():
            if self.config.mode == "P":
                self.config.pmode_orientation_bias_init.instantiate(InitInterface)(self.orientation_bias)
                self.config.pmode_orientation_weight_init.instantiate(InitInterface)(self.orientation_weight)

        for _, m in self.named_children():
            m.reset_parameters(*args, **kwargs)

    @staticmethod
    def _transpose_hxy(*args):
        ret = []
        for arg in args:
            if arg is None:
                ret.append(None)
            else:
                ret.append(arg.transpose(1, 3).transpose(2, 3).reshape(-1, arg.shape[1], arg.shape[2], *arg.shape[4:]))
        return ret

    def _unflatten_orientations(self, orientation_idx, flips, *args):
        ret = []
        for arg in args:
            if arg is None:
                ret.append(None)
            else:
                ret.append(
                    arg.unflatten(
                        3,
                        (
                            len(self.config.orientation_combinations),
                            self.config.num_heads,
                        ),
                    )[:, :, :, orientation_idx].flip(flips)
                )
        return ret

    def _conv_input(self, x, c, part: str):
        if self.config.additional_convolution and part in self.config.convolution_inputs:
            return c
        else:
            return x

    def forward(
        self,
        x,
        initial_state=None,
        return_state: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if initial_state is not None:
            C0_l, C0_t = initial_state
        else:
            C0_l, C0_t = None, None

        # no padding so far for bad shapes
        B, X, Y, D = x.shape
        H = self.config.num_heads
        assert D == self.config.input_dim

        if self.config.additional_convolution:
            c_h = self.convolution(x.reshape(B, X, Y, D))
        else:
            c_h = None
        x_h = x

        q = self.query(self._conv_input(x_h, c_h, "Q"))  # B, X, Y, H, DK, JQ
        if self.config.tie_query_key:
            k = q
        else:
            k = self.key(self._conv_input(x_h, c_h, "K"))  # B, X, Y, H, DK, JK
        v = self.value_(self._conv_input(x_h, c_h, "V"))  # B, X, Y, H, DV, JV

        d = self.direct(self._conv_input(x_h, c_h, "D"))

        if self.config.mode == "P":
            orientation = torch.sigmoid(
                self.orientation_bias
                + self.config.pmode_orientation_scale
                * torch.einsum(
                    "ohx,...x->...oh",
                    self.orientation_weight,
                    self._conv_input(x_h, c_h, "O"),
                )
                / self.config.input_dim
            ).flatten(start_dim=-2)
            s_r = orientation[:, :, :, :, None, None, None] * self.source(self._conv_input(x_h, c_h, "S"))
            s_d = 1.0 - s_r
            t_rl = orientation[:, :, :, :, None, None] * self.transition(self._conv_input(x_h, c_h, "T"))
            t_ru = t_rl
            t_dl = 1 - t_rl
            t_du = 1 - t_rl
            m_l = self.mark(self._conv_input(x_h, c_h, "M"))
            m_u = m_l
        else:
            s_r = self.source_r(self._conv_input(x_h, c_h, "S"))
            s_d = self.source_d(self._conv_input(x_h, c_h, "S"))
            t_rl = self.transition_r(self._conv_input(x_h, c_h, "T"))
            t_du = self.transition_d(self._conv_input(x_h, c_h, "T"))
            t_rd = self.transition_rd(self._conv_input(x_h, c_h, "T"))
            t_ru = None if self.config.dmode_chirality == "R" else t_rd
            t_dl = None if self.config.dmode_chirality == "L" else t_rd
            m_l = self.mark_l(self._conv_input(x_h, c_h, "M"))
            m_u = self.mark_u(self._conv_input(x_h, c_h, "M"))

        outputs = []
        Cs = []
        for orientation_idx, orientation in enumerate(self.config.orientation_combinations):
            flips = []
            if orientation % 2 == 1:
                flips.append(1)
            if (orientation >> 2) % 2 == 1:
                flips.append(2)
            q_flipped = q.flip(flips)
            k_flipped = k.flip(flips)
            v_flipped = v.flip(flips)
            d_flipped = d.flip(flips)

            # take orientation from heads
            (
                s_r_uf,
                s_d_uf,
                t_rl_uf,
                t_du_uf,
                t_dl_uf,
                t_ru_uf,
                m_l_uf,
                m_u_uf,
            ) = self._unflatten_orientations(orientation_idx, flips, s_r, s_d, t_rl, t_du, t_dl, t_ru, m_l, m_u)
            (
                q_flipped,
                k_flipped,
                v_flipped,
                s_r_uf,
                s_d_uf,
                t_rl_uf,
                t_du_uf,
                t_dl_uf,
                t_ru_uf,
                m_l_uf,
                m_u_uf,
                d_uf,
            ) = self._transpose_hxy(
                q_flipped,
                k_flipped,
                v_flipped,
                s_r_uf,
                s_d_uf,
                t_rl_uf,
                t_du_uf,
                t_dl_uf,
                t_ru_uf,
                m_l_uf,
                m_u_uf,
                d_flipped,
            )

            levels = min(self.config.levels, 1 + log2(X - 1), 1 + log2(Y - 1))
            if X % (1 << levels) != 0 or Y % (1 << levels) != 0:
                levels = min(self.config.levels, 1 + log2(X - 1), 1 + log2(Y - 1))
                X_padded = X + ((1 << levels) - X % (1 << levels))
                Y_padded = Y + ((1 << levels) - Y % (1 << levels))

                def _pad(ar):
                    return (
                        torch.cat(
                            (
                                torch.cat(
                                    (
                                        ar,
                                        torch.zeros(
                                            (ar.shape[0], X_padded - X, Y, *ar.shape[3:]),
                                            dtype=ar.dtype,
                                            device=ar.device,
                                        ),
                                    ),
                                    dim=1,
                                ),
                                torch.zeros(
                                    (ar.shape[0], X_padded, Y_padded - Y, *ar.shape[3:]),
                                    dtype=ar.dtype,
                                    device=ar.device,
                                ),
                            ),
                            dim=2,
                        )
                        if ar is not None
                        else None
                    )

                q_flipped = _pad(q_flipped)
                k_flipped = _pad(k_flipped)
                v_flipped = _pad(v_flipped)
                s_r_uf = _pad(s_r_uf)
                s_d_uf = _pad(s_d_uf)
                t_rl_uf = _pad(t_rl_uf)
                t_du_uf = _pad(t_du_uf)
                t_dl_uf = _pad(t_dl_uf)
                t_ru_uf = _pad(t_ru_uf)
                m_l_uf = _pad(m_l_uf)
                m_u_uf = _pad(m_u_uf)
                d_uf = _pad(d_uf)
            else:
                X_padded = X
                Y_padded = Y

            out = pLSTM2D_fwbw(
                q_flipped,
                k_flipped,
                v_flipped,
                s_r_uf,
                s_d_uf,
                t_rl_uf,
                t_du_uf,
                t_dl_uf,
                t_ru_uf,
                m_l_uf,
                m_u_uf,
                d_uf,
                C_initial_left=C0_l,
                C_initial_top=C0_t,
                return_last_C=return_state,
                levels=levels,
            )
            if return_state:
                (
                    out,
                    C_r,
                    C_d,
                ) = out
                Cs.append((C_r, C_d))

            out = out[:, :X, :Y]
            out = out.view([B, H, X, Y, self.config.DV * self.config.JO])
            if self.config.mhnorm:
                outputs.append(self.mhnorm(out).permute(0, 2, 3, 1, 4).flatten(-3, -1).flip(flips))
            else:
                outputs.append(out.permute(0, 2, 3, 1, 4).flatten(-3, -1).flip(flips))

        o = torch.mean(torch.stack(outputs, dim=0), dim=0)

        out = o.reshape(B, X, Y, D)

        if self.config.outprojection:
            out = self.outprojection(out)

        if self.config.additional_passthrough:
            out += self.passthrough(x.reshape(B, X, Y, D))

        if return_state:
            return out.reshape(x.shape), Cs
        else:
            return out.reshape(x.shape)
