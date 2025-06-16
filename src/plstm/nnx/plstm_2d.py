from collections.abc import Callable

try:
    from ..util import log2
    from .util import plus, identity, static_jit
except ImportError:
    import sys
    import os

    sys.path.append(os.path.join(os.path.split(os.path.abspath(__file__))), "..")
    from util import log2
    from jax.util import plus, identity, static_jit
import jax
import jax.numpy as jnp

"""
Naming conventions:
Indices x,y.. refer to positions in the big grid (e.g. up to X//2**level)
Indices a,b.. refer to indices along borders
Indices w,z.. refer to indices within a current (meta-)cell (e.g. up to 2**level)
Indices i,j.. refer to indices for transition internal matrices (in case of more than scalar transitions)
Indices n,m.. refer to batch size, head size etc

"""


@static_jit(static_argnums=(4,))
def transition_matrices_2d(
    T0_rl,
    T0_du,
    T0_dl,
    T0_ru,
    levels: int,
    einsum: Callable[[str, ...], jax.Array] = jnp.einsum,
    acc: Callable[[jax.Array, jax.Array], jax.Array] = plus,
    null_elem: Callable[[jax.Array], jax.Array] = identity,
) -> tuple[list[jax.Array], list[jax.Array], list[jax.Array], list[jax.Array]]:
    """
    T0: base transition matrices going from one adjacent point to another adjacent one
        [N, X, Y, A, A, J, J]
    >>> from .util import allclose_verbose
    >>> bool(allclose_verbose(
    ...     transition_matrices_2d(
    ...         jnp.ones([1, 2, 2, 1, 1]), jnp.ones([1, 2, 2, 1, 1]),
    ...         jnp.ones([1, 2, 2, 1, 1]), jnp.ones([1, 2, 2, 1, 1]), 1)[1][1],
    ...     jnp.array([[[[[[[1.]], [[0.]]], [[[2.]], [[1.]]]]]]])))
    True
    """
    B, X, Y, J, _ = T0_rl.shape
    T_rl = [T0_rl[:, :, :, None, None]]
    T_du = [T0_du[:, :, :, None, None]]
    T_dl = [T0_dl[:, :, :, None, None]] if T0_dl is not None else None
    T_ru = [T0_ru[:, :, :, None, None]] if T0_ru is not None else None

    for i in range(log2(min(T0_du.shape[1], T0_du.shape[2], (1 << levels)))):
        T_rl.append(
            jnp.concatenate(
                [
                    jnp.concatenate(
                        [
                            einsum(
                                "nxyabij,nxybcjk->nxyacik",
                                T_rl[i][:, 1::2, 0:-1:2],
                                T_rl[i][:, 0:-1:2, 0:-1:2],
                            ),
                            null_elem(jnp.zeros_like(T_rl[i][:, 1::2, 0:-1:2])),
                        ],
                        axis=4,
                    ),
                    jnp.concatenate(
                        [
                            (
                                (
                                    acc(
                                        einsum(
                                            "nxyabij,nxybcjk,nxycdkl->nxyadil",
                                            T_ru[i][:, 1::2, 1::2],
                                            T_dl[i][:, 1::2, 0:-1:2],
                                            T_rl[i][:, 0:-1:2, 0:-1:2],
                                        ),
                                        einsum(
                                            "nxyabij,nxybcjk,nxycdkl->nxyadil",
                                            T_rl[i][:, 1::2, 1::2],
                                            T_ru[i][:, 0:-1:2, 1::2],
                                            T_dl[i][:, 0:-1:2, 0:-1:2],
                                        ),
                                    )
                                )
                                if T_ru is not None and T_dl is not None
                                else null_elem(jnp.zeros_like(T_rl[i][:, 1::2, 0:-1:2]))
                            ),
                            einsum(
                                "nxyabij,nxybcjk->nxyacik",
                                T_rl[i][:, 1::2, 1::2],
                                T_rl[i][:, 0:-1:2, 1::2],
                            ),
                        ],
                        axis=4,
                    ),
                ],
                axis=3,
            )
        )
        T_du.append(
            jnp.concatenate(
                [
                    jnp.concatenate(
                        [
                            einsum(
                                "nxyabij,nxybcjk->nxyacik",
                                T_du[i][:, 0:-1:2, 1::2],
                                T_du[i][:, 0:-1:2, 0:-1:2],
                            ),
                            null_elem(jnp.zeros_like(T_rl[i][:, 1::2, 0:-1:2])),
                        ],
                        axis=4,
                    ),
                    jnp.concatenate(
                        [
                            (
                                (
                                    acc(
                                        einsum(
                                            "nxyabij,nxybcjk,nxycdkl->nxyadil",
                                            T_dl[i][:, 1::2, 1::2],
                                            T_ru[i][:, 0:-1:2, 1::2],
                                            T_du[i][:, 0:-1:2, 0:-1:2],
                                        ),
                                        einsum(
                                            "nxyabij,nxybcjk,nxycdkl->nxyadil",
                                            T_du[i][:, 1::2, 1::2],
                                            T_dl[i][:, 1::2, 0:-1:2],
                                            T_ru[i][:, 0:-1:2, 0:-1:2],
                                        ),
                                    )
                                )
                                if T_ru is not None and T_dl is not None
                                else null_elem(jnp.zeros_like(T_rl[i][:, 1::2, 0:-1:2]))
                            ),
                            einsum(
                                "nxyabij,nxybcjk->nxyacik",
                                T_du[i][:, 1::2, 1::2],
                                T_du[i][:, 1::2, 0:-1:2],
                            ),
                        ],
                        axis=4,
                    ),
                ],
                axis=3,
            )
        )
        if T_ru is not None:
            T_ru.append(
                jnp.concatenate(
                    [
                        jnp.concatenate(
                            [
                                einsum(
                                    "nxyabij,nxybcjk->nxyacik",
                                    T_rl[i][:, 1::2, 0:-1:2],
                                    T_ru[i][:, 0:-1:2, 0:-1:2],
                                ),
                                T_ru[i][:, 1::2, 0:-1:2],
                            ],
                            axis=4,
                        ),
                        jnp.concatenate(
                            [
                                acc(
                                    einsum(
                                        "nxyabij,nxybcjk,nxycdkl->nxyadil",
                                        T_rl[i][:, 1::2, 1::2],
                                        T_ru[i][:, 0:-1:2, 1::2],
                                        T_du[i][:, 0:-1:2, 0:-1:2],
                                    ),
                                    (
                                        einsum(
                                            "nxyabij,nxybcjk,nxycdkl->nxyadil",
                                            T_ru[i][:, 1::2, 1::2],
                                            T_dl[i][:, 1::2, 0:-1:2],
                                            T_ru[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_dl is not None
                                        else null_elem(jnp.zeros_like(T_du[i][:, 1::2, 1::2]))
                                    ),
                                ),
                                einsum(
                                    "nxyabij,nxybcjk->nxyacik",
                                    T_ru[i][:, 1::2, 1::2],
                                    T_du[i][:, 1::2, 0:-1:2],
                                ),
                            ],
                            axis=4,
                        ),
                    ],
                    axis=3,
                )
            )
        if T_dl is not None:
            T_dl.append(
                jnp.concatenate(
                    [
                        jnp.concatenate(
                            [
                                einsum(
                                    "nxyabij,nxybcjk->nxyacik",
                                    T_du[i][:, 0:-1:2, 1::2],
                                    T_dl[i][:, 0:-1:2, 0:-1:2],
                                ),
                                T_dl[i][:, 0:-1:2, 1::2],
                            ],
                            axis=4,
                        ),
                        jnp.concatenate(
                            [
                                acc(
                                    einsum(
                                        "nxyabij,nxybcjk,nxycdkl->nxyadil",
                                        T_du[i][:, 1::2, 1::2],
                                        T_dl[i][:, 1::2, 0:-1:2],
                                        T_rl[i][:, 0:-1:2, 0:-1:2],
                                    ),
                                    (
                                        einsum(
                                            "nxyabij,nxybcjk,nxycdkl->nxyadil",
                                            T_dl[i][:, 1::2, 1::2],
                                            T_ru[i][:, 0:-1:2, 1::2],
                                            T_dl[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_ru is not None
                                        else null_elem(jnp.zeros_like(T_du[i][:, 0:-1:2, 0:-1:2]))
                                    ),
                                ),
                                einsum(
                                    "nxyabij,nxybcjk->nxyacik",
                                    T_dl[i][:, 1::2, 1::2],
                                    T_rl[i][:, 0:-1:2, 1::2],
                                ),
                            ],
                            axis=4,
                        ),
                    ],
                    axis=3,
                )
            )
    return T_rl, T_du, T_dl, T_ru


@jax.jit
def _source_reshape(X: jax.Array):
    return X[:, : (X.shape[1] >> 1) << 1, :, : (X.shape[3] >> 1) << 1 :].reshape(
        X.shape[0],
        X.shape[1] >> 1,
        X.shape[2] << 1,
        X.shape[3] >> 1,
        X.shape[4] << 1,
        *X.shape[5:],
    )


@static_jit(static_argnums=(6,))
def source_matrices_2d(
    S0_r: jax.Array,
    S0_d: jax.Array,
    T_rl: list[jax.Array],
    T_du: list[jax.Array],
    T_dl: list[jax.Array],
    T_ru: list[jax.Array],
    levels: int,
    einsum: Callable[[str, ...], jax.Array] = jnp.einsum,
    acc: Callable[[jax.Array, jax.Array], jax.Array] = plus,
    null_elem: Callable[[jax.Array], jax.Array] = identity,
) -> tuple[list[jax.Array], list[jax.Array]]:
    S_r = [S0_r[:, :, None, :, None, None]]
    S_d = [S0_d[:, :, None, :, None, None]]
    for i in range(log2(min(S0_r.shape[1], S0_r.shape[2], (1 << levels)))):
        Sr = _source_reshape(S_r[i])
        Sd = _source_reshape(S_d[i])
        S_r.append(
            jnp.concatenate(
                [
                    jnp.concatenate(
                        [
                            jnp.concatenate(
                                [
                                    einsum(
                                        "nxyabij,nxwyzbjkl->nxwyzaikl",
                                        T_rl[i][:, 1::2, 0:-1:2],
                                        Sr[:, :, : (1 << i), :, : (1 << i), :],
                                    ),
                                    acc(
                                        einsum(
                                            "nxyabij,nxybcjk,nxwyzcklm->nxwyzailm",
                                            T_ru[i][:, 1::2, 1::2],
                                            T_dl[i][:, 1::2, 0:-1:2],
                                            Sr[:, :, : (1 << i), :, : (1 << i), :],
                                        )
                                        if T_dl is not None and T_ru is not None
                                        else null_elem(jnp.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                        einsum(
                                            "nxyabij,nxybcjk,nxwyzcklm->nxwyzailm",
                                            T_rl[i][:, 1::2, 1::2],
                                            T_ru[i][:, 0:-1:2, 1::2],
                                            Sd[:, :, : (1 << i), :, : (1 << i), :],
                                        )
                                        if T_ru is not None
                                        else null_elem(jnp.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    ),
                                ],
                                axis=5,
                            ),
                            jnp.concatenate(
                                [
                                    null_elem(jnp.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    einsum(
                                        "nxyabij,nxwyzbjkl->nxwyzaikl",
                                        T_rl[i][:, 1::2, 1::2],
                                        Sr[:, :, : (1 << i), :, (1 << i) :, :],
                                    ),
                                ],
                                axis=5,
                            ),
                        ],
                        axis=4,
                    ),
                    jnp.concatenate(
                        [
                            jnp.concatenate(
                                [
                                    Sr[:, :, (1 << i) :, :, : (1 << i), :],
                                    (
                                        einsum(
                                            "nxyabij,nxwyzbjkl->nxwyzaikl",
                                            T_ru[i][:, 1::2, 1::2],
                                            Sd[:, :, (1 << i) :, :, : (1 << i), :],
                                        )
                                        if T_ru is not None
                                        else null_elem(jnp.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)]))
                                    ),
                                ],
                                axis=5,
                            ),
                            jnp.concatenate(
                                [
                                    null_elem(jnp.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    Sr[:, :, (1 << i) :, :, (1 << i) :, :],
                                ],
                                axis=5,
                            ),
                        ],
                        axis=4,
                    ),
                ],
                axis=2,
            )
        )
        S_d.append(
            jnp.concatenate(
                [
                    jnp.concatenate(
                        [
                            jnp.concatenate(
                                [
                                    einsum(
                                        "nxyabij,nxwyzbjkl->nxwyzaikl",
                                        T_du[i][:, 0:-1:2, 1::2],
                                        Sd[:, :, : (1 << i), :, : (1 << i), :],
                                    ),
                                    acc(
                                        einsum(
                                            "nxyabij,nxybcjk,nxwyzcklm->nxwyzailm",
                                            T_dl[i][:, 1::2, 1::2],
                                            T_ru[i][:, 0:-1:2, 1::2],
                                            Sd[:, :, : (1 << i), :, : (1 << i), :],
                                        )
                                        if T_dl is not None and T_ru is not None
                                        else null_elem(
                                            jnp.zeros_like(
                                                Sd[
                                                    :,
                                                    :,
                                                    : (1 << i),
                                                    :,
                                                    : (1 << i),
                                                    :,
                                                ]
                                            )
                                        ),
                                        einsum(
                                            "nxyabij,nxybcjk,nxwyzcklm->nxwyzailm",
                                            T_du[i][:, 1::2, 1::2],
                                            T_dl[i][:, 1::2, 0:-1:2],
                                            Sr[:, :, : (1 << i), :, : (1 << i), :],
                                        )
                                        if T_dl is not None
                                        else null_elem(jnp.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    ),
                                ],
                                axis=5,
                            ),
                            jnp.concatenate(
                                [
                                    Sd[:, :, : (1 << i), :, (1 << i) :, :],
                                    (
                                        einsum(
                                            "nxyabij,nxwyzbjkl->nxwyzaikl",
                                            T_dl[i][:, 1::2, 1::2],
                                            Sr[:, :, : (1 << i), :, (1 << i) :, :],
                                        )
                                        if T_dl is not None
                                        else null_elem(jnp.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)]))
                                    ),
                                ],
                                axis=5,
                            ),
                        ],
                        axis=4,
                    ),
                    jnp.concatenate(
                        [
                            jnp.concatenate(
                                [
                                    null_elem(jnp.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    einsum(
                                        "nxyabij,nxwyzbjkl->nxwyzaikl",
                                        T_du[i][:, 1::2, 1::2],
                                        Sd[:, :, (1 << i) :, :, : (1 << i), :],
                                    ),
                                ],
                                axis=5,
                            ),
                            jnp.concatenate(
                                [
                                    null_elem(jnp.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    Sd[:, :, (1 << i) :, :, (1 << i) :, :],
                                ],
                                axis=5,
                            ),
                        ],
                        axis=4,
                    ),
                ],
                axis=2,
            )
        )
    return S_r, S_d


@static_jit(static_argnums=(6,))
def mark_matrices_2d(
    M0_l: jax.Array,
    M0_u: jax.Array,
    T_rl: list[jax.Array],
    T_du: list[jax.Array],
    T_dl: list[jax.Array],
    T_ru: list[jax.Array],
    levels: int,
    einsum: Callable[[str, ...], jax.Array] = jnp.einsum,
    acc: Callable[[jax.Array, jax.Array], jax.Array] = plus,
    null_elem: Callable[[jax.Array], jax.Array] = identity,
) -> tuple[list[jax.Array], list[jax.Array]]:
    M_l = [M0_l[:, :, None, :, None, None]]
    M_u = [M0_u[:, :, None, :, None, None]]
    for i in range(log2(min(M0_l.shape[1], M0_u.shape[2], 1 << levels))):
        Mu = _source_reshape(M_u[i])
        Ml = _source_reshape(M_l[i])
        M_l.append(
            jnp.concatenate(
                [
                    jnp.concatenate(
                        [
                            jnp.concatenate(
                                [
                                    Ml[:, :, : (1 << i), :, : (1 << i)],
                                    null_elem(jnp.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                ],
                                axis=5,
                            ),
                            jnp.concatenate(
                                [
                                    (
                                        einsum(
                                            "nxwyzaijk,nxyabkl->nxwyzbijl",
                                            Mu[:, :, : (1 << i), :, (1 << i) :, :],
                                            T_dl[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_dl is not None
                                        else null_elem(jnp.zeros_like(Mu[:, :, : (1 << i), :, : (1 << i)]))
                                    ),
                                    Ml[:, :, : (1 << i), :, (1 << i) :, :],
                                ],
                                axis=5,
                            ),
                        ],
                        axis=4,
                    ),
                    jnp.concatenate(
                        [
                            jnp.concatenate(
                                [
                                    einsum(
                                        "nxwyzaijk,nxyabkl->nxwyzbijl",
                                        Ml[:, :, (1 << i) :, :, : (1 << i), :],
                                        T_rl[i][:, 0:-1:2, 0:-1:2],
                                    ),
                                    null_elem(jnp.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                ],
                                axis=5,
                            ),
                            jnp.concatenate(
                                [
                                    acc(
                                        einsum(
                                            "nxwyzaijk,nxyabkl,nxybclm->nxwyzcijm",
                                            Ml[:, :, (1 << i) :, :, (1 << i) :, :],
                                            T_ru[i][:, 0:-1:2, 1::2],
                                            T_dl[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_dl is not None and T_ru is not None
                                        else null_elem(jnp.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                        einsum(
                                            "nxwyzaijk,nxyabkl,nxybclm->nxwyzcijm",
                                            Mu[:, :, (1 << i) :, :, (1 << i) :, :],
                                            T_dl[i][:, 1::2, 0:-1:2],
                                            T_rl[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_dl is not None
                                        else null_elem(jnp.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                    ),
                                    einsum(
                                        "nxwyzaijk,nxyabkl->nxwyzbijl",
                                        Ml[:, :, (1 << i) :, :, (1 << i) :, :],
                                        T_rl[i][:, 0:-1:2, 1::2],
                                    ),
                                ],
                                axis=5,
                            ),
                        ],
                        axis=4,
                    ),
                ],
                axis=2,
            )
        )
        M_u.append(
            jnp.concatenate(
                [
                    jnp.concatenate(
                        [
                            jnp.concatenate(
                                [
                                    Mu[:, :, : (1 << i), :, : (1 << i)],
                                    null_elem(jnp.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                ],
                                axis=5,
                            ),
                            jnp.concatenate(
                                [
                                    einsum(
                                        "nxwyzaijk,nxyabkl->nxwyzbijl",
                                        Mu[:, :, : (1 << i), :, (1 << i) :, :],
                                        T_du[i][:, 0:-1:2, 0:-1:2],
                                    ),
                                    null_elem(jnp.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                ],
                                axis=5,
                            ),
                        ],
                        axis=4,
                    ),
                    jnp.concatenate(
                        [
                            jnp.concatenate(
                                [
                                    (
                                        einsum(
                                            "nxwyzaijk,nxyabkl->nxwyzbijl",
                                            Ml[:, :, (1 << i) :, :, : (1 << i), :],
                                            T_ru[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_ru is not None
                                        else null_elem(jnp.zeros_like(Mu[:, :, : (1 << i), :, : (1 << i)]))
                                    ),
                                    Mu[:, :, (1 << i) :, :, : (1 << i), :],
                                ],
                                axis=5,
                            ),
                            jnp.concatenate(
                                [
                                    (
                                        einsum(
                                            "nxwyzaijk,nxyabkl,nxybclm->nxwyzcijm",
                                            Ml[:, :, (1 << i) :, :, (1 << i) :, :],
                                            T_ru[i][:, 0:-1:2, 1::2],
                                            T_du[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_ru is not None
                                        else null_elem(jnp.zeros_like(Mu[:, :, : (1 << i), :, : (1 << i)]))
                                    )
                                    + (
                                        einsum(
                                            "nxwyzaijk,nxyabkl,nxybclm->nxwyzcijm",
                                            Mu[:, :, (1 << i) :, :, (1 << i) :, :],
                                            T_dl[i][:, 1::2, 0:-1:2],
                                            T_ru[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_dl is not None and T_ru is not None
                                        else null_elem(jnp.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)]))
                                    ),
                                    einsum(
                                        "nxwyzaijk,nxyabkl->nxwyzbijl",
                                        Mu[:, :, (1 << i) :, :, (1 << i) :, :],
                                        T_du[i][:, 1::2, 0:-1:2],
                                    ),
                                ],
                                axis=5,
                            ),
                        ],
                        axis=4,
                    ),
                ],
                axis=2,
            )
        )
    return M_l, M_u


@static_jit(static_argnums=(8,))
def gating_matrices_2d(
    S_r: list[jax.Array],
    S_d: list[jax.Array],
    M_l: list[jax.Array],
    M_u: list[jax.Array],
    T_rl: list[jax.Array],
    T_du: list[jax.Array],
    T_dl: list[jax.Array],
    T_ru: list[jax.Array],
    levels: int,
    einsum: Callable[[str, ...], jax.Array] = jnp.einsum,
    acc: Callable[[jax.Array, jax.Array], jax.Array] = plus,
    null_elem: Callable[[jax.Array], jax.Array] = identity,
):
    B, X, Y, JK, JV, JO, JQ = (
        S_r[0].shape[0],
        S_r[0].shape[1],
        S_r[0].shape[3],
        S_r[0].shape[7],
        S_r[0].shape[8],
        M_l[0].shape[6],
        M_l[0].shape[7],
    )

    C = min((1 << log2(X)), (1 << log2(Y)), (1 << levels))
    Xc = X // C
    Yc = Y // C
    G_main = jnp.zeros([B, Xc, C, C, Yc, C, C, JO, JQ, JK, JV], dtype=S_r[0].dtype)
    # Aggregate top left part and levels in between
    for i in range(log2(C)):  # range(levels - 1):
        g = jnp.zeros(
            [
                B,
                Xc,
                C >> (i + 1),
                1 << (i + 1),
                1 << (i + 1),
                Yc,
                C >> (i + 1),
                1 << (i + 1),
                1 << (i + 1),
                JO,
                JQ,
                JK,
                JV,
            ],
            dtype=S_r[0].dtype,
        )
        g_part_shape = (
            B,
            Xc,
            C >> (i + 1),
            1 << i,
            1 << i,
            Yc,
            C >> (i + 1),
            1 << i,
            1 << i,
            JO,
            JQ,
            JK,
            JV,
        )
        g = g.at[:, :, :, (1 << i) :, : (1 << i), :, :, : (1 << i), : (1 << i)].set(
            einsum(
                "nxwyzaijk,nxuyvaklm->nxwuyzvijlm",
                M_l[i][:, 1::2, :, 0:-1:2],
                S_r[i][:, 0:-1:2, :, 0:-1:2],
            ).reshape(*g_part_shape)
        )
        g = g.at[:, :, :, : (1 << i), : (1 << i), :, :, (1 << i) :, : (1 << i)].set(
            einsum(
                "nxwyzaijk,nxuyvaklm->nxwuyzvijlm",
                M_u[i][:, 0:-1:2, :, 1::2],
                S_d[i][:, 0:-1:2, :, 0:-1:2],
            ).reshape(*g_part_shape)
        )
        g = g.at[:, :, :, (1 << i) :, : (1 << i), :, :, (1 << i) :, : (1 << i)].set(
            (
                (
                    einsum(
                        "nxwyzaijk,nxyabkl,nxuyvblmh->nxwuyzvijmh",
                        M_u[i][:, 1::2, :, 1::2],
                        T_dl[i][:, 1::2, 0:-1:2],
                        S_r[i][:, 0:-1:2, :, 0:-1:2],
                    )
                    if T_dl is not None
                    else 0
                )
                + (
                    einsum(
                        "nxwyzaijk,nxyabkl,nxuyvblmh->nxwuyzvijmh",
                        M_l[i][:, 1::2, :, 1::2],
                        T_ru[i][:, 0:-1:2, 1::2],
                        S_d[i][:, 0:-1:2, :, 0:-1:2],
                    )
                    if T_ru is not None
                    else 0
                )
            ).reshape(*g_part_shape)
        )
        g = g.at[:, :, :, (1 << i) :, (1 << i) :, :, :, (1 << i) :, : (1 << i)].set(
            einsum(
                "nxwyzaijk,nxuyvaklm->nxwuyzvijlm",
                M_u[i][:, 1::2, :, 1::2],
                S_d[i][:, 1::2, :, 0:-1:2],
            ).reshape(*g_part_shape)
        )
        g = g.at[:, :, :, (1 << i) :, : (1 << i), :, :, (1 << i) :, (1 << i) :].add(
            einsum(
                "nxwyzaijk,nxuyvaklm->nxwuyzvijlm",
                M_l[i][:, 1::2, :, 1::2],
                S_r[i][:, 0:-1:2, :, 1::2],
            ).reshape(*g_part_shape)
        )
        G_main = G_main.at[
            :,
            :,
            : g.shape[2] * g.shape[3],
            : g.shape[2] * g.shape[3],
            :,
            : g.shape[6] * g.shape[7],
            : g.shape[6] * g.shape[7],
        ].add(
            (
                g[:, :, :, :, None, :, :, :, :, None, :, :, :, :]
                * (
                    jnp.eye(g.shape[2], dtype=g.dtype)[
                        None,
                        None,
                        :,
                        None,
                        :,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ]
                    * jnp.eye(g.shape[6], dtype=g.dtype)[
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        :,
                        None,
                        :,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ]
                )
            ).reshape(
                B,
                g.shape[1],
                g.shape[2] * g.shape[3],
                g.shape[2] * g.shape[3],
                g.shape[5],
                g.shape[6] * g.shape[7],
                g.shape[6] * g.shape[7],
                g.shape[-4],
                g.shape[-3],
                g.shape[-2],
                g.shape[-1],
            )
        )

    return G_main


@static_jit(static_argnums=(8,))
def stmg_matrices(
    S0_r: jax.Array,
    S0_d: jax.Array,
    T0_rl: jax.Array,
    T0_du: jax.Array,
    T0_dl: jax.Array,
    T0_ru: jax.Array,
    M0_l: jax.Array,
    M0_u: jax.Array,
    levels: int = 20,
    einsum: Callable[[str, ...], jax.Array] = jnp.einsum,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    T_rl, T_du, T_dl, T_ru = transition_matrices_2d(T0_rl, T0_du, T0_dl, T0_ru, levels=levels)
    S_r, S_d = source_matrices_2d(S0_r, S0_d, T_rl, T_du, T_dl, T_ru, levels=levels)
    M_l, M_u = mark_matrices_2d(M0_l, M0_u, T_rl, T_du, T_dl, T_ru, levels=levels)
    G = gating_matrices_2d(S_r, S_d, M_l, M_u, T_rl, T_du, T_dl, T_ru, levels=levels)
    return (
        S_r[-1],
        S_d[-1],
        T_rl[-1],
        T_du[-1],
        T_dl[-1] if T_dl is not None else None,
        T_ru[-1] if T_ru is not None else None,
        M_l[-1],
        M_u[-1],
        G,
    )


def pLSTM2D_jax(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    S0_r: jax.Array,
    S0_d: jax.Array,
    T0_rl: jax.Array,
    T0_du: jax.Array,
    T0_dl: jax.Array,
    T0_ru: jax.Array,
    M0_l: jax.Array,
    M0_u: jax.Array,
    D0: jax.Array,
    C_initial_left: jax.Array | None = None,
    C_initial_top: jax.Array | None = None,
    levels: int = 16,
    # TODO: the second two args are dummy currently and should be implemented via jax.remat
    recompute_G: bool = False,
    recompute_C: bool = False,
    use_initial_C: bool = False,
    return_last_C: bool = False,
    QK_scale: float | None = None,
):
    DB, MX, MY, DK, DV, JT, JK, JV, JQ, JO = (
        Q.shape[0],
        Q.shape[1],
        Q.shape[2],
        Q.shape[3],
        V.shape[3],
        S0_r.shape[3],
        S0_r.shape[4],
        S0_r.shape[5],
        M0_l.shape[4],
        M0_l.shape[3],
    )
    if QK_scale is None:
        QK_scale = DK ** (-1 / 2)
    BP = 1 << levels
    NX = MX // BP
    NY = MY // BP
    D = D0.reshape(-1, NX, BP, 1, NY, BP, 1, JO, JQ, JK, JV)

    S_r, S_d, T_rl, T_du, T_dl, T_ru, M_l, M_u, G_nodirect = stmg_matrices(
        S0_r,
        S0_d,
        T0_rl,
        T0_du,
        T0_dl,
        T0_ru,
        M0_l,
        M0_u,
        levels=levels,
    )

    G = G_nodirect + (
        D
        * jnp.eye(BP, dtype=D.dtype)[None, None, :, :, None, None, None, None, None, None, None]
        * jnp.eye(BP, dtype=D.dtype)[None, None, None, None, None, :, :, None, None, None, None]
    )
    Q = Q.reshape(DB, NX, BP, NY, BP, DK, JQ)
    K = K.reshape(DB, NX, BP, NY, BP, DK, JK)
    V = V.reshape(DB, NX, BP, NY, BP, DV, JV)

    if C_initial_left is not None or C_initial_top is not None or NY > 1 or NX > 1 or return_last_C:
        inter_chunk_Skv_r = jnp.einsum(
            "nxuywaijk,nxuywdj,nxuywvk->nxyadvi",
            S_r,
            K,
            V,
        )
        inter_chunk_Skv_d = jnp.einsum(
            "nxuywaijk,nxuywdj,nxuywvk->nxyadvi",
            S_d,
            K,
            V,
        )

        C_r = [[None for _ in range(NY + 1)] for _ in range(NX + 1)]
        C_d = [[None for _ in range(NY + 1)] for _ in range(NX + 1)]
        C_r[0][0] = jnp.zeros([DB, BP, DK, DV, JT], dtype=Q.dtype)
        C_d[0][0] = jnp.zeros([DB, BP, DK, DV, JT], dtype=Q.dtype)
        for ny in range(NY):
            C_d[0][ny + 1] = jnp.zeros([DB, BP, DK, DV, JT], dtype=Q.dtype)
        for nx in range(NX):
            C_r[nx + 1][0] = jnp.zeros([DB, BP, DK, DV, JT], dtype=Q.dtype)

        if C_initial_left is not None:
            for ny in range(NY):
                C_r[0][ny + 1] = C_initial_left[:, ny]
        else:
            for ny in range(NY):
                C_r[0][ny + 1] = jnp.zeros([DB, BP, DK, DV, JT], dtype=Q.dtype)
        if C_initial_top is not None:
            for nx in range(NX):
                C_d[nx + 1][0] = C_initial_top[:, nx]
        else:
            for nx in range(NX):
                C_d[nx + 1][0] = jnp.zeros([DB, BP, DK, DV, JT], dtype=Q.dtype)

        for j in range(0, NX + NY - 1):
            for i in range(min(j + 1, NX, NY, NX + NY - j - 1)):
                if j < NX and j < NY:
                    x = i + 1
                    y = j - i + 1
                elif j >= NX and j >= NY:
                    x = j - NY + i + 2
                    y = NY - i
                elif j >= NX:
                    x = NX - i
                    y = j - NX + i + 2
                elif j >= NY:
                    x = j - NY + i + 2
                    y = NY - i
                C_r[x][y] = (
                    jnp.einsum(
                        "nabij,nbdvj->nadvi",
                        T_rl[:, x - 1, y - 1],
                        C_r[x - 1][y],
                    )
                    + (
                        jnp.einsum(
                            "nabij,nbdvj->nadvi",
                            T_ru[:, x - 1, y - 1],
                            C_d[x][y - 1],
                        )
                        if T_ru is not None
                        else 0
                    )
                    + inter_chunk_Skv_r[:, x - 1, y - 1]
                )
                C_d[x][y] = (
                    jnp.einsum(
                        "nabij,nbdvj->nadvi",
                        T_du[:, x - 1, y - 1],
                        C_d[x][y - 1],
                    )
                    + inter_chunk_Skv_d[:, x - 1, y - 1]
                    + (
                        jnp.einsum(
                            "nabij,nbdvj->nadvi",
                            T_dl[:, x - 1, y - 1],
                            C_r[x - 1][y],
                        )
                        if T_dl is not None
                        else 0
                    )
                )

        C_r = jnp.stack([jnp.stack(Cy, axis=0) for Cy in C_r], axis=0).transpose(2, 0, 1, 3, 4, 5, 6)
        C_d = jnp.stack([jnp.stack(Cy, axis=0) for Cy in C_d], axis=0).transpose(2, 0, 1, 3, 4, 5, 6)

        inter_chunk = jnp.einsum("nxwyudj,nxwyuaijk,nxyadvk->nxwyuvi", Q * QK_scale, M_l, C_r[:, :-1, 1:]) + jnp.einsum(
            "nxwyudj,nxwyuaijk,nxyadvk->nxwyuvi", Q * QK_scale, M_u, C_d[:, 1:, :-1]
        )
    else:
        inter_chunk = 0

    intra_chunk = jnp.einsum(
        "nxwyzdj,nxwuyztijkl,nxuytdk,nxuytvl->nxwyzvi",
        Q * QK_scale,
        G,
        K,
        V,
    )

    if return_last_C:
        return (
            (inter_chunk + intra_chunk).reshape(DB, MX, MY, DV, JO),
            C_r[:, -1, :],
            C_d[:, :, -1],
        )
    else:
        return (inter_chunk + intra_chunk).reshape(DB, MX, MY, DV, JO)


# no recurrent states!
def pLSTM2D_parallel_fused_jax(
    # non-fused wrt orientation
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    # fused versions with all orientations rd, ld, ru, lu
    S0_r: jax.Array,  # [B, O, MX, MY, JT, JK, JV]
    S0_d: jax.Array,  # [B, O, MX, MY, JT, JK, JV]
    T0_rl: jax.Array,  # [B, O, MX, MY, JT, JT]
    T0_du: jax.Array,  # [B, O, MX, MY, JT, JT]
    T0_dl: jax.Array,  # [B, O, MX, MY, JT, JT]
    T0_ru: jax.Array,  # [B, O, MX, MY, JT, JT]
    M0_l: jax.Array,  # [B, O, MX, MY, JO, JQ, JT]
    M0_u: jax.Array,  # [B, O, MX, MY, JO, JQ, JT]
    # non-fused wrt orientation
    D0: jax.Array,
    levels: int = 16,
    recompute_G: bool = False,
    return_G: bool = False,
    QK_scale: float | None = None,
):
    DB, MX, MY, DK, DV, JT, JK, JV, JQ, JO = (
        Q.shape[0],
        S0_r.shape[2],
        S0_r.shape[3],
        Q.shape[3],
        V.shape[3],
        S0_r.shape[4],
        S0_r.shape[5],
        S0_r.shape[6],
        M0_l.shape[5],
        M0_l.shape[4],
    )
    UX, UY = Q.shape[1], Q.shape[2]

    if QK_scale is None:
        QK_scale = DK ** (-1 / 2)
    BP = 1 << levels
    NX = MX // BP
    NY = MY // BP
    assert NX == 1
    assert NY == 1
    D = D0.reshape(-1, BP, 1, BP, 1, JO, JQ, JK, JV)

    def flips(A: jnp.ndarray | None) -> jnp.ndarray | None:
        return jnp.stack(
            [
                A[:, 0],
                jnp.flip(A[:, 1], axis=1),
                jnp.flip(A[:, 2], axis=2),
                jnp.flip(jnp.flip(A[:, 3], axis=1), axis=2),
            ],
            axis=1,
        )

    S0f_r, S0f_d = (
        flips(S0_r).reshape(DB * 4, MX, MY, JT, JK, JV),
        flips(S0_d).reshape(DB * 4, MX, MY, JT, JK, JV),
    )
    T0f_rl, T0f_du, T0f_dl, T0f_ru = (
        flips(T0_rl).reshape(DB * 4, MX, MY, JT, JT),
        flips(T0_du).reshape(DB * 4, MX, MY, JT, JT),
        flips(T0_dl).reshape(DB * 4, MX, MY, JT, JT) if T0_dl is not None else None,
        flips(T0_ru).reshape(DB * 4, MX, MY, JT, JT) if T0_ru is not None else None,
    )
    M0f_l, M0f_u = (
        flips(M0_l).reshape(DB * 4, MX, MY, JO, JQ, JT),
        flips(M0_u).reshape(DB * 4, MX, MY, JO, JQ, JT),
    )

    # S_r, S_d, _, _, _, _, _, _,
    S_r, S_d, T_rl, T_du, T_dl, T_ru, M_l, M_u, G_nodirect = stmg_matrices(
        S0f_r,
        S0f_d,
        T0f_rl,
        T0f_du,
        T0f_dl,
        T0f_ru,
        M0f_l,
        M0f_u,
        levels=levels,
    )

    G_nodirect_unflipped = G_nodirect.reshape(DB, 4, MX, MX, MY, MY, JO, JQ, JK, JV)
    # sum up all the four orientations
    G_nodirect_allorientations = (
        G_nodirect_unflipped[:, 0]
        + jnp.flip(jnp.flip(G_nodirect_unflipped[:, 1], axis=1), axis=2)
        + jnp.flip(jnp.flip(G_nodirect_unflipped[:, 2], axis=3), axis=4)
        + jnp.flip(jnp.flip(jnp.flip(jnp.flip(G_nodirect_unflipped[:, 3], axis=1), axis=2), axis=3), axis=4)
    )

    G = G_nodirect_allorientations + (
        D
        * jnp.eye(BP, dtype=D.dtype)[None, :, :, None, None, None, None, None, None]
        * jnp.eye(BP, dtype=D.dtype)[None, None, None, :, :, None, None, None, None]
    )

    # jax.debug.print("Gmax: {G_max}", G_max=jnp.max(G))

    Q = Q.reshape(DB, UX, UY, DK, JQ)
    K = K.reshape(DB, UX, UY, DK, JK)
    V = V.reshape(DB, UX, UY, DV, JV)

    intra_chunk = jnp.einsum(
        "nxydj,nxwyzijkl,nwzdk,nwzvl->nxyvi",
        Q * QK_scale,
        G[:, :UX, :UX, :UY, :UY],
        K,
        V,
    )

    if return_G:
        return intra_chunk.reshape(DB, UX, UY, DV, JO), G[:, :UX, :UX, :UY, :UY]
    else:
        return intra_chunk.reshape(DB, UX, UY, DV, JO)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)

    DB, MX, MY, DHQK, DHHV, JQ, JK, JV, JT, JO = (3, 8, 8, 32, 32, 1, 1, 1, 1, 1)

    Q, K, V = (
        jnp.ones([DB, MX, MY, DHQK, JQ]),
        jnp.ones([DB, MX, MY, DHQK, JK]),
        jnp.ones([DB, MX, MY, DHHV, JV]),
    )

    S0, T0, M0, D0 = (
        jnp.ones([DB, MX, MY, JT, JK, JV]),
        jnp.ones([DB, MX, MY, JT, JT]),
        jnp.ones([DB, MX, MY, JO, JQ, JT]),
        jnp.ones([DB, MX, MY, JO, JQ, JK, JV]),
    )

    res = pLSTM2D_jax(Q, K, V, S0, S0, T0, T0, 0.0 * T0, T0, M0, M0, D0, levels=3)
    print(res)

    res = pLSTM2D_jax(Q, K, V, S0, S0, T0, T0, None, T0, M0, M0, D0, levels=3)
    print(res)
