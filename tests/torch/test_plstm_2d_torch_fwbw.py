import pytest
import torch

from plstm.torch.plstm_2d import pLSTM2D_fwbw, pLSTM2D_torch
from plstm.torch.test_utils import check_forward


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_plstm_2d_fwbw(levels, request):
    B = 1
    X = 8
    Y = 8
    DHQK = 1
    DHHV = 1
    JQ = 1
    JT = 1
    JV = 1
    JK = 1
    JO = 1

    phi0 = 0.5
    phi1 = 1 - phi0

    Q = 1.0 + 0.0 * torch.randn([B, X, Y, DHQK, JQ]) / DHQK
    K = 1.0 + 0.0 * torch.randn([B, X, Y, DHQK, JK])
    V = 1.0 + 0.0 * torch.randn([B, X, Y, DHHV, JV])
    S0_r = phi0 * (1.0 + 0.0 * torch.randn([B, X, Y, JT, JK, JV]))
    S0_d = phi1 * (1.0 + 0.0 * torch.randn([B, X, Y, JT, JK, JV]))
    T00 = torch.eye(JT)[None, None, None, :, :] * torch.ones([B, X, Y, 1, 1])
    T0_rl = phi0 * T00 + 0.0 * torch.randn_like(T00)
    T0_du = phi1 * T00 + 0.0 * torch.randn_like(T00)
    T0_dl = phi1 * T00 + 0.0 * torch.randn_like(T00)
    T0_ru = phi0 * T00 + 0.0 * torch.randn_like(T00)
    M0_l = 1.0 + 0.0 * torch.randn([B, X, Y, JO, JQ, JT])
    M0_u = 1.0 + 0.0 * torch.randn([B, X, Y, JO, JQ, JT])
    D0 = 1 + 0.0 * torch.randn([B, X, Y, JO, JQ, JK, JV])

    check_forward(
        lambda q, k, v, s0r, s0d, t0rl, t0du, t0dl, t0ru, m0l, m0u, d0: pLSTM2D_torch(
            q, k, v, s0r, s0d, t0rl, t0du, t0dl, t0ru, m0l, m0u, d0, levels=levels
        ),
        lambda q, k, v, s0r, s0d, t0rl, t0du, t0dl, t0ru, m0l, m0u, d0: pLSTM2D_fwbw(
            q, k, v, s0r, s0d, t0rl, t0du, t0dl, t0ru, m0l, m0u, d0, levels=levels
        ),
        (Q, K, V, S0_r, S0_d, T0_rl, T0_du, T0_dl, T0_ru, M0_l, M0_u, D0),
        verbose=True,
    )
