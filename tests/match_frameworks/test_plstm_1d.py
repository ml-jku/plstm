import pytest
import jax.numpy as jnp
import torch
import numpy as np

from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools
from plstm.nnx.plstm_1d import pLSTM1D_jax
from plstm.torch.plstm_1d import pLSTM1D_torch


def convert_to_numpy(x):
    """Convert JAX or PyTorch tensor to numpy array."""
    if isinstance(x, jnp.ndarray):
        return np.array(x)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_plstm_1d_basic(levels, request):
    """Test basic functionality."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Test dimensions
    DB, DT, DHQK, DHHV = 2, 8, 4, 4  # Small sizes for testing
    JQ, JK, JV, JT, JO = 1, 1, 1, 4, 1

    # Create test inputs
    rng = np.random.RandomState(42)

    # JAX inputs
    Q_np = rng.randn(DB, DT, DHQK, JQ)
    K_np = rng.randn(DB, DT, DHQK, JK)
    V_np = rng.randn(DB, DT, DHHV, JV)
    S0_np = rng.randn(DB, DT, JT, JK, JV)
    T0_np = rng.randn(DB, DT, JT, JT)
    M0_np = rng.randn(DB, DT, JO, JQ, JT)
    D0_np = rng.randn(DB, DT, JO, JQ, JK, JV)

    Q_jax = jnp.array(Q_np)
    K_jax = jnp.array(K_np)
    V_jax = jnp.array(V_np)
    S0_jax = jnp.array(S0_np)
    T0_jax = jnp.array(T0_np)
    M0_jax = jnp.array(M0_np)
    D0_jax = jnp.array(D0_np)

    # PyTorch inputs
    Q_torch = torch.tensor(Q_np).float()
    K_torch = torch.tensor(K_np).float()
    V_torch = torch.tensor(V_np).float()
    S0_torch = torch.tensor(S0_np).float()
    T0_torch = torch.tensor(T0_np).float()
    M0_torch = torch.tensor(M0_np).float()
    D0_torch = torch.tensor(D0_np).float()

    # Run both implementations
    out_jax = pLSTM1D_jax(Q_jax, K_jax, V_jax, S0_jax, T0_jax, M0_jax, D0_jax, levels=levels)
    out_torch = pLSTM1D_torch(Q_torch, K_torch, V_torch, S0_torch, T0_torch, M0_torch, D0_torch, levels=levels)

    # Convert outputs to numpy for comparison
    out_jax_np = convert_to_numpy(out_jax)
    out_torch_np = convert_to_numpy(out_torch)

    # Compare outputs
    assert_allclose_with_plot(
        out_jax_np, out_torch_np, rtol=1.5e-2, atol=1e-4, base_path=f"{test_name}_{next(counter)}"
    )


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_plstm_1d_with_initial_state(levels, request):
    """Test with initial state (C_initial and E_initial)."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Test dimensions
    DB, DT, DHQK, DHHV = 2, 8, 4, 4
    JQ, JK, JV, JT, JO = 1, 1, 1, 4, 1

    # Create test inputs
    rng = np.random.RandomState(42)

    # JAX inputs
    Q_np = rng.randn(DB, DT, DHQK, JQ)
    K_np = rng.randn(DB, DT, DHQK, JK)
    V_np = rng.randn(DB, DT, DHHV, JV)
    S0_np = rng.randn(DB, DT, JT, JK, JV)
    T0_np = rng.randn(DB, DT, JT, JT)
    M0_np = rng.randn(DB, DT, JO, JQ, JT)
    D0_np = rng.randn(DB, DT, JO, JQ, JK, JV)
    C_initial_np = rng.randn(DB, DHQK, DHHV, JT)

    Q_jax = jnp.array(Q_np)
    K_jax = jnp.array(K_np)
    V_jax = jnp.array(V_np)
    S0_jax = jnp.array(S0_np)
    T0_jax = jnp.array(T0_np)
    M0_jax = jnp.array(M0_np)
    D0_jax = jnp.array(D0_np)
    C_initial_jax = jnp.array(C_initial_np)

    # PyTorch inputs
    Q_torch = torch.tensor(Q_np).float()
    K_torch = torch.tensor(K_np).float()
    V_torch = torch.tensor(V_np).float()
    S0_torch = torch.tensor(S0_np).float()
    T0_torch = torch.tensor(T0_np).float()
    M0_torch = torch.tensor(M0_np).float()
    D0_torch = torch.tensor(D0_np).float()
    C_initial_torch = torch.tensor(C_initial_np).float()

    # Run both implementations with return_last_C=True
    out_jax, C_last_jax = pLSTM1D_jax(
        Q_jax,
        K_jax,
        V_jax,
        S0_jax,
        T0_jax,
        M0_jax,
        D0_jax,
        C_initial=C_initial_jax,
        levels=levels,
        return_last_C=True,
    )
    out_torch, C_last_torch = pLSTM1D_torch(
        Q_torch,
        K_torch,
        V_torch,
        S0_torch,
        T0_torch,
        M0_torch,
        D0_torch,
        C_initial=C_initial_torch,
        levels=levels,
        return_last_C=True,
    )

    # Convert outputs to numpy for comparison
    out_jax_np = convert_to_numpy(out_jax)
    out_torch_np = convert_to_numpy(out_torch)
    C_last_jax_np = convert_to_numpy(C_last_jax)
    C_last_torch_np = convert_to_numpy(C_last_torch)

    # Compare outputs
    assert_allclose_with_plot(out_jax_np, out_torch_np, rtol=2e-2, atol=1e-2, base_path=f"{test_name}_{next(counter)}")
    assert_allclose_with_plot(
        C_last_jax_np, C_last_torch_np, rtol=2e-2, atol=1e-2, base_path=f"{test_name}_{next(counter)}"
    )
