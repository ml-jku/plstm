import pytest
import jax
import jax.numpy as jnp
import torch
import numpy as np
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools

from plstm.nnx.plstm_2d import pLSTM2D_jax
from plstm.torch.plstm_2d import pLSTM2D_torch


def convert_to_numpy(x):
    """Convert JAX or PyTorch tensor to numpy array."""
    if isinstance(x, jax.Array):
        return np.array(x)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_plstm_2d_with_directional_transitions(levels: int, request):
    """Test with all directional transition matrices (_d, _r, _l, _u, _lr, _ld,
    _ur, _ud)."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Test dimensions
    DB, MX, MY = 2, 8, 8
    JQ, JK, JV, JT, JO = 1, 1, 1, 4, 1
    BP = 1 << levels

    # Create test inputs
    rng = np.random.RandomState(42)

    # JAX inputs
    Q_np = rng.randn(DB, MX, MY, JQ)
    K_np = rng.randn(DB, MX, MY, JK)
    V_np = rng.randn(DB, MX, MY, JV)
    S0_r_np = rng.randn(DB, MX, MY, JT, JK, JV)  # right direction
    S0_d_np = rng.randn(DB, MX, MY, JT, JK, JV)  # down direction
    T0_rl_np = rng.randn(DB, MX, MY, JT, JT)  # right-to-left
    T0_du_np = rng.randn(DB, MX, MY, JT, JT)  # down-to-up
    T0_dl_np = rng.randn(DB, MX, MY, JT, JT)  # diagonal down-left
    T0_ru_np = rng.randn(DB, MX, MY, JT, JT)  # diagonal right-up
    # P mode normalization
    T0_rl_np = T0_rl_np / (np.abs(T0_rl_np) + np.abs(T0_ru_np))
    T0_ru_np = T0_ru_np / (np.abs(T0_rl_np) + np.abs(T0_ru_np))
    T0_du_np = T0_du_np / (np.abs(T0_du_np) + np.abs(T0_dl_np))
    T0_dl_np = T0_dl_np / (np.abs(T0_du_np) + np.abs(T0_dl_np))

    M0_l_np = rng.randn(DB, MX, MY, JO, JQ, JT)  # left direction
    M0_u_np = rng.randn(DB, MX, MY, JO, JQ, JT)  # up direction
    D0_np = rng.randn(DB, MX, MY, JO, JQ, JK, JV)

    # Initial states
    C_initial_left_np = rng.randn(DB, MY // BP, BP, JK, JV, JT)  # Initial left border state
    C_initial_top_np = rng.randn(DB, MX // BP, BP, JK, JV, JT)  # Initial top border state

    Q_jax = jnp.array(Q_np)
    K_jax = jnp.array(K_np)
    V_jax = jnp.array(V_np)
    S0_r_jax = jnp.array(S0_r_np)
    S0_d_jax = jnp.array(S0_d_np)
    T0_rl_jax = jnp.array(T0_rl_np)
    T0_du_jax = jnp.array(T0_du_np)
    T0_dl_jax = jnp.array(T0_dl_np)
    T0_ru_jax = jnp.array(T0_ru_np)
    M0_l_jax = jnp.array(M0_l_np)
    M0_u_jax = jnp.array(M0_u_np)
    D0_jax = jnp.array(D0_np)
    C_initial_left_jax = jnp.array(C_initial_left_np)
    C_initial_top_jax = jnp.array(C_initial_top_np)

    # PyTorch inputs
    Q_torch = torch.tensor(Q_np).float()
    K_torch = torch.tensor(K_np).float()
    V_torch = torch.tensor(V_np).float()
    S0_r_torch = torch.tensor(S0_r_np).float()
    S0_d_torch = torch.tensor(S0_d_np).float()
    T0_rl_torch = torch.tensor(T0_rl_np).float()
    T0_du_torch = torch.tensor(T0_du_np).float()
    T0_dl_torch = torch.tensor(T0_dl_np).float()
    T0_ru_torch = torch.tensor(T0_ru_np).float()
    M0_l_torch = torch.tensor(M0_l_np).float()
    M0_u_torch = torch.tensor(M0_u_np).float()
    D0_torch = torch.tensor(D0_np).float()
    C_initial_left_torch = torch.tensor(C_initial_left_np).float()
    C_initial_top_torch = torch.tensor(C_initial_top_np).float()

    # Test without stabilization and initial states
    out_jax = pLSTM2D_jax(
        Q_jax,
        K_jax,
        V_jax,
        S0_r_jax,
        S0_d_jax,
        T0_rl_jax,
        T0_du_jax,
        T0_dl_jax,
        T0_ru_jax,
        M0_l_jax,
        M0_u_jax,
        D0_jax,
        levels=levels,
    )
    out_torch = pLSTM2D_torch(
        Q_torch,
        K_torch,
        V_torch,
        S0_r_torch,
        S0_d_torch,
        T0_rl_torch,
        T0_du_torch,
        T0_dl_torch,
        T0_ru_torch,
        M0_l_torch,
        M0_u_torch,
        D0_torch,
        levels=levels,
    )

    # Compare outputs without stabilization and initial states
    out_jax_np = convert_to_numpy(out_jax)
    out_torch_np = convert_to_numpy(out_torch)
    assert_allclose_with_plot(
        out_jax_np, out_torch_np, rtol=1.2e-1, atol=1e-2, base_path=f"{test_name}_{next(counter)}"
    )

    # Test with stabilization and initial states
    out_jax_stab = pLSTM2D_jax(
        Q_jax,
        K_jax,
        V_jax,
        S0_r_jax,
        S0_d_jax,
        T0_rl_jax,
        T0_du_jax,
        T0_dl_jax,
        T0_ru_jax,
        M0_l_jax,
        M0_u_jax,
        D0_jax,
        C_initial_left=C_initial_left_jax,
        C_initial_top=C_initial_top_jax,
        levels=levels,
    )
    out_torch_stab = pLSTM2D_torch(
        Q_torch,
        K_torch,
        V_torch,
        S0_r_torch,
        S0_d_torch,
        T0_rl_torch,
        T0_du_torch,
        T0_dl_torch,
        T0_ru_torch,
        M0_l_torch,
        M0_u_torch,
        D0_torch,
        C_initial_left=C_initial_left_torch,
        C_initial_top=C_initial_top_torch,
        levels=levels,
    )

    # Compare outputs with stabilization and initial states
    out_jax_stab_np = convert_to_numpy(out_jax_stab)
    out_torch_stab_np = convert_to_numpy(out_torch_stab)
    assert_allclose_with_plot(
        out_jax_stab_np, out_torch_stab_np, rtol=1.2e-1, atol=1e-2, base_path=f"{test_name}_{next(counter)}"
    )
