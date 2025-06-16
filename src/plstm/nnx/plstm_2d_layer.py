import jax
from plstm.nnx_dummy import nnx
import jax.numpy as jnp
from compoconf import register
from ..config.plstm_2d_layer import pLSTM2DLayerConfig
from ..util import log2
from .norm import NormInterface
from .initialization import InitInterface

from .query_key_value import QueryLayer, KeyLayer, ValueLayer
from .source_mark_layer import SourceLayer, MarkLayer, DirectLayer
from .longrange_transition_layer import (
    LongRangeTransitionLayer,
)
from .passthrough import PassthroughLayer
from .convolution import Convolution2DLayer
from .plstm_2d import pLSTM2D_jax
from .util import weight_einsum
from .interfaces import ResidualModule

from .dtype import str_dtype_to_jax


@register
class pLSTM2DLayer(ResidualModule):
    config: pLSTM2DLayerConfig

    def __init__(self, config: pLSTM2DLayerConfig, *, rngs: nnx.Rngs):
        nnx.Module.__init__(self)
        self.config = config

        if config.additional_convolution:
            self.convolution = Convolution2DLayer(config.convolution, rngs=rngs)

        if config.mode == "P":
            orientation_bias_shape = [len(config.orientation_combinations), config.num_heads]
            self.orientation_bias = nnx.Param(
                config.pmode_orientation_bias_init.instantiate(InitInterface)(
                    rngs.params(),
                    orientation_bias_shape,
                    dtype=str_dtype_to_jax(config.param_dtype),
                )
            )
            orientation_weight_shape = [
                len(config.orientation_combinations),
                config.num_heads,
                config.input_dim,
            ]
            self.orientation_weight = nnx.Param(
                config.pmode_orientation_weight_init.instantiate(InitInterface)(
                    rngs.params(), orientation_weight_shape, dtype=str_dtype_to_jax(config.param_dtype)
                )
            )

            self.source = SourceLayer(config.source, rngs=rngs)
            self.transition = LongRangeTransitionLayer(config.transition, rngs=rngs)
            self.mark = MarkLayer(config.mark, rngs=rngs)

        elif config.mode == "D":
            self.source_r = SourceLayer(config.source, rngs=rngs)
            self.source_d = SourceLayer(config.source, rngs=rngs)

            self.transition_r = LongRangeTransitionLayer(config.transition, rngs=rngs)
            self.transition_d = LongRangeTransitionLayer(config.transition, rngs=rngs)
            self.transition_rd = LongRangeTransitionLayer(config.transition, rngs=rngs)

            self.mark_l = MarkLayer(config.mark, rngs=rngs)
            self.mark_u = MarkLayer(config.mark, rngs=rngs)

        self.direct = DirectLayer(config.direct, rngs=rngs)
        self.query = QueryLayer(config.query, rngs=rngs)
        if not config.tie_query_key:
            self.key = KeyLayer(config.key, rngs=rngs)
        self.value_ = ValueLayer(config.value, rngs=rngs)

        if config.mhnorm is not None:
            self.mhnorm = config.mhnorm.instantiate(NormInterface, rngs=rngs)

        if config.outprojection:
            self.outprojection = nnx.Linear(
                in_features=config.num_heads * config.JO * config.DV,
                out_features=config.input_dim,
                use_bias=config.outprojection_bias,
                kernel_init=config.outprojection_weight_init.instantiate(InitInterface),
                bias_init=config.outprojection_bias_init.instantiate(InitInterface),
                dtype=str_dtype_to_jax(config.dtype),
                param_dtype=str_dtype_to_jax(config.param_dtype),
                rngs=rngs,
            )

        if config.additional_passthrough:
            self.passthrough = PassthroughLayer(config.passthrough, rngs=rngs)

    def _gating_function(self, x):
        if self.config.gating_function == "silu":
            return jax.nn.silu(x)
        else:
            raise ValueError("Bad gating function")

    @staticmethod
    def _transpose_hxy(*args):
        ret = []
        for arg in args:
            if arg is None:
                ret.append(None)
            else:
                ret.append(
                    arg.transpose(0, 3, 1, 2, *range(4, arg.ndim)).reshape(
                        -1, arg.shape[1], arg.shape[2], *arg.shape[4:]
                    )
                )
        return ret

    def _unflatten_orientations(self, orientation_idx, flips, *args):
        ret = []
        for arg in args:
            if arg is None:
                ret.append(None)
            else:
                arg_unflat = arg.reshape(
                    *arg.shape[:3], len(self.config.orientation_combinations), self.config.num_heads, *arg.shape[4:]
                )[:, :, :, orientation_idx]
                for flip in flips:
                    arg_unflat = jnp.flip(arg_unflat, axis=flip)
                ret.append(arg_unflat)
        return ret

    def _conv_input(self, x, c, part: str):
        if self.config.additional_convolution and part in self.config.convolution_inputs:
            return c
        else:
            return x

    def __call__(
        self,
        x,
        initial_state=None,
        return_state: bool = False,
        **kwargs,
    ) -> jax.Array:
        if initial_state is not None:
            C0_l, C0_t = initial_state
        else:
            C0_l, C0_t = None, None

        B, X, Y, D = x.shape
        H, E = self.config.num_heads, self.config.input_dim // self.config.num_heads

        if self.config.additional_convolution:
            c_h = self.convolution(x)
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
            orientation = jax.nn.sigmoid(
                self.orientation_bias
                + self.config.pmode_orientation_scale
                * weight_einsum("ohx,...x->...oh", self.orientation_weight, self._conv_input(x_h, c_h, "O"))
                / self.config.input_dim
            ).reshape(*x_h.shape[:-1], -1)
            s_r = orientation[..., None, None, None] * self.source(self._conv_input(x_h, c_h, "S"))
            s_d = 1.0 - s_r
            t_rl = orientation[..., None, None] * self.transition(self._conv_input(x_h, c_h, "T"))
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
            q_flipped = jnp.flip(q, flips)
            k_flipped = jnp.flip(k, flips)
            v_flipped = jnp.flip(v, flips)
            d_flipped = jnp.flip(d, flips)

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
                        jnp.pad(
                            ar, ((0, 0), (0, X_padded - X), (0, Y_padded - Y), *((0, 0) for _ in range(ar.ndim - 3)))
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

            out = pLSTM2D_jax(
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
                out, C_r, C_d, E_r, E_d = out
                Cs.append((C_r, C_d))
            out = out[:, :X, :Y]
            out = out.reshape([B, H, X, Y, self.config.DV * self.config.JO])
            if self.config.mhnorm is not None:
                out = self.mhnorm(out)
            out = out.transpose(0, 2, 3, 1, 4).reshape(B, X, Y, -1)
            for flip in flips:
                out = jnp.flip(out, axis=flip)
            outputs.append(out)

        o = jnp.mean(jnp.stack(outputs, axis=0), axis=0)
        out = o.reshape(B, X, Y, H * E)

        if self.config.outprojection:
            out = self.outprojection(out)

        if self.config.additional_passthrough:
            out = out + self.passthrough(x.reshape(B, X, Y, H * E))

        if return_state:
            return out.reshape(x.shape), Cs
        else:
            return out.reshape(x.shape)
