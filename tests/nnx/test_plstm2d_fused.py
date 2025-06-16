import jax
import jax.numpy as jnp
import torch
import numpy as np
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools

from plstm.nnx.plstm_2d import pLSTM2D_jax, pLSTM2D_parallel_fused_jax


def convert_to_numpy(x):
    """Convert JAX or PyTorch tensor to numpy array."""
    if isinstance(x, jax.Array):
        return np.array(x)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def test_plstm_2d_fused(request):
    """Test with all directional transition matrices (_d, _r, _l, _u, _lr, _ld,
    _ur, _ud)."""
    # Test dimensions
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()

    DB, MX, MY = 2, 8, 8
    JQ, JK, JV, JT, JO = 1, 1, 1, 1, 1
    DK, DV = 8, 8
    levels = 3

    # Create test inputsq
    rng = np.random.RandomState(42)
    # JAX inputs
    Q_np = rng.randn(DB, MX, MY, DK * JQ)
    K_np = rng.randn(DB, MX, MY, DK * JK)
    V_np = rng.randn(DB, MX, MY, DV * JV)
    S0_r_np = rng.randn(DB, 4, MX, MY, JT, JK, JV)  # right direction
    S0_d_np = rng.randn(DB, 4, MX, MY, JT, JK, JV)  # down direction
    T0_rl_np = rng.uniform(0.0, 1.0, size=(DB, 4, MX, MY, JT, JT))  # right-to-left
    T0_du_np = rng.uniform(0.0, 1.0, size=(DB, 4, MX, MY, JT, JT))  # diagonal right-up
    T0_dl_np = 1.0 - T0_rl_np
    T0_ru_np = 1.0 - T0_du_np
    M0_l_np = rng.randn(DB, 4, MX, MY, JO, JQ, JT)  # left direction
    M0_u_np = rng.randn(DB, 4, MX, MY, JO, JQ, JT)  # up direction
    D0_np = rng.randn(DB, MX, MY, JO, JQ, JK, JV)

    # Initial states
    # Q_np[:1] = Q_np[:1]
    # K_np[:1] = K_np[:1]
    # V_np[:1] = V_np[:1]

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

    # Test without stabilization and initial states
    out_jax_directions = []
    for i in range(4):

        def flip(x, i):
            if i == 0:
                return x
            if i == 1:
                return jnp.flip(x, axis=1)
            if i == 2:
                return jnp.flip(x, axis=2)
            if i == 3:
                return jnp.flip(jnp.flip(x, axis=1), axis=2)
            else:
                raise ValueError

        out_jax_directions.append(
            pLSTM2D_jax(
                flip(Q_jax, i),
                flip(K_jax, i),
                flip(V_jax, i),
                flip(S0_r_jax[:, i], i),
                flip(S0_d_jax[:, i], i),
                flip(T0_rl_jax[:, i], i),
                flip(T0_du_jax[:, i], i),
                flip(T0_dl_jax[:, i], i),
                flip(T0_ru_jax[:, i], i),
                flip(M0_l_jax[:, i], i),
                flip(M0_u_jax[:, i], i),
                flip(0.25 * D0_jax, i),
                levels=levels,
            )
        )

    out_jax_baseline = (
        out_jax_directions[0]
        + jnp.flip(out_jax_directions[1], axis=1)
        + jnp.flip(out_jax_directions[2], axis=2)
        + jnp.flip(jnp.flip(out_jax_directions[3], axis=1), axis=2)
    )

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

    # Compare outputs without stabilization and initial states
    out_jax_np = convert_to_numpy(out_jax_baseline)
    out_jaxf_np = convert_to_numpy(out_jax)
    assert_allclose_with_plot(
        out_jaxf_np,
        out_jax_np,
        rtol=1e-2,
        atol=1e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert np.max(out_jax_np) < 1e3
