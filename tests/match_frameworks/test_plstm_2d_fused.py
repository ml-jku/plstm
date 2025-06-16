import jax
import jax.numpy as jnp
import torch
import numpy as np
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools

from plstm.nnx.plstm_2d import pLSTM2D_parallel_fused_jax
from plstm.torch.plstm_2d_fused import pLSTM2D_parallel_fused_torch


def convert_to_numpy(x):
    """Convert JAX or PyTorch tensor to numpy array."""
    if isinstance(x, jax.Array):
        return np.array(x)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def test_plstm_2d_fused(request):
    """Test the fused implementation of pLSTM2D with all orientations."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Test dimensions
    DB, MX, MY = 2, 8, 8
    DK, DV = 32, 32
    JQ, JK, JV, JT, JO = 1, 1, 1, 4, 1
    levels = 3

    # Create test inputs
    rng = np.random.RandomState(42)

    # Create inputs
    Q_np = rng.randn(DB, MX, MY, DK, JQ)
    K_np = rng.randn(DB, MX, MY, DK, JK)
    V_np = rng.randn(DB, MX, MY, DV, JV)

    # Create orientation-aware inputs (4 orientations)
    S0_r_np = 0.01 * rng.randn(DB, 4, MX, MY, JT, JK, JV)  # right direction
    S0_d_np = 0.01 * rng.randn(DB, 4, MX, MY, JT, JK, JV)  # down direction
    T0_rl_np = rng.randn(DB, 4, MX, MY, JT, JT)  # right-to-left
    T0_du_np = rng.randn(DB, 4, MX, MY, JT, JT)  # down-to-up
    T0_dl_np = rng.randn(DB, 4, MX, MY, JT, JT)  # diagonal down-left
    T0_ru_np = rng.randn(DB, 4, MX, MY, JT, JT)  # diagonal right-up

    # P mode normalization
    T0_rl_np = T0_rl_np / (np.abs(T0_rl_np) + np.abs(T0_ru_np) + 1e-6)
    T0_ru_np = T0_ru_np / (np.abs(T0_rl_np) + np.abs(T0_ru_np) + 1e-6)
    T0_du_np = T0_du_np / (np.abs(T0_du_np) + np.abs(T0_dl_np) + 1e-6)
    T0_dl_np = T0_dl_np / (np.abs(T0_du_np) + np.abs(T0_dl_np) + 1e-6)

    M0_l_np = rng.randn(DB, 4, MX, MY, JO, JQ, JT)  # left direction
    M0_u_np = rng.randn(DB, 4, MX, MY, JO, JQ, JT)  # up direction
    D0_np = rng.randn(DB, MX, MY, JO, JQ, JK, JV)

    # Convert to JAX arrays
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

    # Convert to PyTorch tensors
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

    # Test without return_G
    out_jax = pLSTM2D_parallel_fused_jax(
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
    out_torch = pLSTM2D_parallel_fused_torch(
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

    # Compare outputs
    out_jax_np = convert_to_numpy(out_jax)
    out_torch_np = convert_to_numpy(out_torch)
    assert_allclose_with_plot(out_jax_np, out_torch_np, rtol=0.1, atol=3e-2, base_path=f"{test_name}_{next(counter)}")

    # Test with return_G
    out_jax, G_jax = pLSTM2D_parallel_fused_jax(
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
        return_G=True,
    )
    out_torch, G_torch = pLSTM2D_parallel_fused_torch(
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
        return_G=True,
    )

    # Compare outputs with return_G
    out_jax_np = convert_to_numpy(out_jax)
    out_torch_np = convert_to_numpy(out_torch)
    G_jax_np = convert_to_numpy(G_jax)
    G_torch_np = convert_to_numpy(G_torch)

    assert_allclose_with_plot(out_jax_np, out_torch_np, rtol=0.4, atol=1e-1, base_path=f"{test_name}_{next(counter)}")
    assert_allclose_with_plot(G_jax_np, G_torch_np, rtol=0.4, atol=1e-1, base_path=f"{test_name}_{next(counter)}")


def test_plstm_2d_fused_with_none_transitions(request):
    """Test the fused implementation with None for some transition matrices."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Test dimensions
    DB, MX, MY = 2, 8, 8
    DK, DV = 32, 32
    JQ, JK, JV, JT, JO = 1, 1, 1, 4, 1
    levels = 3

    # Create test inputs
    rng = np.random.RandomState(42)

    # Create inputs
    Q_np = rng.randn(DB, MX, MY, DK, JQ)
    K_np = rng.randn(DB, MX, MY, DK, JK)
    V_np = rng.randn(DB, MX, MY, DV, JV)

    # Create orientation-aware inputs (4 orientations)
    S0_r_np = 0.01 * rng.randn(DB, 4, MX, MY, JT, JK, JV)
    S0_d_np = 0.01 * rng.randn(DB, 4, MX, MY, JT, JK, JV)
    T0_rl_np = np.clip(rng.randn(DB, 4, MX, MY, JT, JT), 0.0, 1.0)
    T0_du_np = np.clip(rng.randn(DB, 4, MX, MY, JT, JT), 0.0, 1.0)
    T0_dl_np = np.clip(rng.randn(DB, 4, MX, MY, JT, JT), 0.0, 1.0)
    M0_l_np = rng.randn(DB, 4, MX, MY, JO, JQ, JT)
    M0_u_np = rng.randn(DB, 4, MX, MY, JO, JQ, JT)
    D0_np = rng.randn(DB, MX, MY, JO, JQ, JK, JV)

    # Convert to JAX arrays
    Q_jax = jnp.array(Q_np)
    K_jax = jnp.array(K_np)
    V_jax = jnp.array(V_np)
    S0_r_jax = jnp.array(S0_r_np)
    S0_d_jax = jnp.array(S0_d_np)
    T0_rl_jax = jnp.array(T0_rl_np)
    T0_du_jax = jnp.array(T0_du_np)
    T0_dl_jax = jnp.array(T0_dl_np)
    M0_l_jax = jnp.array(M0_l_np)
    M0_u_jax = jnp.array(M0_u_np)
    D0_jax = jnp.array(D0_np)

    # Convert to PyTorch tensors
    Q_torch = torch.tensor(Q_np).float()
    K_torch = torch.tensor(K_np).float()
    V_torch = torch.tensor(V_np).float()
    S0_r_torch = torch.tensor(S0_r_np).float()
    S0_d_torch = torch.tensor(S0_d_np).float()
    T0_rl_torch = torch.tensor(T0_rl_np).float()
    T0_du_torch = torch.tensor(T0_du_np).float()
    T0_dl_torch = torch.tensor(T0_dl_np).float()
    M0_l_torch = torch.tensor(M0_l_np).float()
    M0_u_torch = torch.tensor(M0_u_np).float()
    D0_torch = torch.tensor(D0_np).float()

    # Test with None for T0_dl and T0_ru
    out_jax = pLSTM2D_parallel_fused_jax(
        Q_jax,
        K_jax,
        V_jax,
        S0_r_jax,
        S0_d_jax,
        T0_rl_jax,
        T0_du_jax,
        T0_dl_jax,
        None,  # T0_ru is None
        M0_l_jax,
        M0_u_jax,
        D0_jax,
        levels=levels,
    )
    out_torch = pLSTM2D_parallel_fused_torch(
        Q_torch,
        K_torch,
        V_torch,
        S0_r_torch,
        S0_d_torch,
        T0_rl_torch,
        T0_du_torch,
        T0_dl_torch,
        None,  # T0_ru is None
        M0_l_torch,
        M0_u_torch,
        D0_torch,
        levels=levels,
    )

    # Compare outputs
    out_jax_np = convert_to_numpy(out_jax)
    out_torch_np = convert_to_numpy(out_torch)
    assert_allclose_with_plot(out_jax_np, out_torch_np, rtol=0.1, atol=3e-2, base_path=f"{test_name}_{next(counter)}")
