from collections.abc import Callable

try:
    from ..util import log2
    from .util import identity, static_jit
except ImportError:
    import sys
    import os

    sys.path.append(os.path.join(os.path.split(os.path.abspath(__file__))[0], ".."))
    from util import log2

    sys.path.append(os.path.join(os.path.split(os.path.abspath(__file__))[0], "."))
    from util import identity, static_jit
import jax
import jax.numpy as jnp
from .dtype import DTYPES


@static_jit(static_argnums=(1,))
def transition_matrices(
    T0: jax.Array, top_level: int, einsum: Callable[[str, ...], jax.Array] = jnp.einsum
) -> list[jax.Array]:
    """
    >>> from .util import allclose_verbose
    >>> bool(allclose_verbose(
    ...     transition_matrices(0.5*jnp.ones([8])[None, :, None, None], 3)[0],
    ...     jnp.array([[[[0.5000]],[[0.5000]],[[0.5000]],[[0.5000]],[[0.5000]],[[0.5000]],[[0.5000]],[[0.5000]]]])))
    True
    >>> bool(allclose_verbose(
    ...     transition_matrices(1.+jnp.arange(4.)[None, :, None, None], 2)[1],
    ...     jnp.array([[[[2.]], [[12.]]]])))
    True
    >>> bool(allclose_verbose(
    ...     transition_matrices(0.5*jnp.ones([8])[None, :, None, None], 3)[1],
    ...     jnp.array([[[[0.2500]],[[0.2500]],[[0.2500]],[[0.2500]]]])))
    True
    >>> bool(allclose_verbose(
    ...     transition_matrices(0.5*jnp.ones([8])[None, :, None, None], 3)[2],
    ...     jnp.array([[[[0.0625]], [[0.0625]]]])))
    True
    >>> bool(allclose_verbose(
    ...     transition_matrices(0.5*jnp.ones([8])[None, :, None, None], 3)[3],
    ...     jnp.array([[[[0.00390625]]]])))
    True
    >>> transition_matrices(0.5*jnp.ones([1, 4, 1, 1]), 2)
    [Array([[[[0.5]],
    <BLANKLINE>
            [[0.5]],
    <BLANKLINE>
            [[0.5]],
    <BLANKLINE>
            [[0.5]]]], dtype=float32), Array([[[[0.25]],
    <BLANKLINE>
            [[0.25]]]], dtype=float32), Array([[[[0.0625]]]], dtype=float32)]
    """
    T = [T0]
    for i in range(top_level):
        T.append(
            einsum("nxij,nxjk->nxik", T[i][:, 1::2], T[i][:, 0:-1:2]),
        )
    return T


@static_jit(static_argnums=(2,))
def source_matrices(S0, T, top_level: int, einsum: Callable[[str, ...], jax.Array] = jnp.einsum) -> list[jax.Array]:
    """
    >>> from .util import allclose_verbose
    >>> bool(allclose_verbose(
    ...      source_matrices(
    ...         1.+jnp.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*jnp.ones([1, 4, 1, 1]), 2), 2)[0],
    ...     jnp.array([[[[[[1.]]]], [[[[2.]]]], [[[[3.]]]], [[[[4.]]]]]])))
    True
    >>> bool(allclose_verbose(
    ...     source_matrices(1+jnp.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*jnp.ones(4)[None, :, None, None], 2), 2)[1],
    ...     jnp.array([[[[[[0.5000]]], [[[2.0000]]]], [[[[1.5000]]], [[[4.0000]]]]]])))
    True
    >>> bool(allclose_verbose(
    ...     source_matrices(1+jnp.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*jnp.ones(4)[None, :, None, None], 2), 2)[2],
    ...     jnp.array([[[[[[0.1250]]], [[[0.5000]]], [[[1.5000]]], [[[4.0000]]]]]])))
    True
    """
    S = [S0[:, :, None, :]]
    for i in range(top_level):
        S.append(
            jnp.concatenate(
                [
                    einsum("nxij,nxajkl->nxaikl", T[i][:, 1::2], S[i][:, 0:-1:2, :]),
                    S[i][:, 1::2, :],
                ],
                axis=2,
            )
        )
    return S


@static_jit(static_argnums=(2,))
def mark_matrices(
    M0,
    T,
    top_level: int,
    einsum: Callable[[str, ...], jax.Array] = jnp.einsum,
) -> list[jax.Array]:
    """
    >>> from .util import allclose_verbose
    >>> bool(allclose_verbose(
    ...     mark_matrices(
    ...         1.+jnp.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*jnp.ones(4)[None, :, None, None], 2), 2)[0],
    ...     jnp.array([[[[[[1.]]]],[[[[2.]]]],[[[[3.]]]],[[[[4.]]]]]])))
    True
    >>> bool(allclose_verbose(
    ...     mark_matrices(
    ...         1+jnp.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*jnp.ones(4)[None, :, None, None], 2), 2)[1],
    ...     jnp.array([[[[[[1.]]], [[[1.]]]],
    ...                   [[[[3.]]], [[[2.]]]]]])))
    True
    >>> bool(allclose_verbose(
    ...     mark_matrices(
    ...         1+jnp.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*jnp.ones(4)[None, :, None, None], 2), 2)[2],
    ...     jnp.array([[[[[[1.0000]]], [[[1.0000]]], [[[0.7500]]], [[[0.5000]]]]]])))
    True
    """
    M = [M0[:, :, None, :]]
    for i in range(top_level):
        M.append(
            jnp.concatenate(
                [
                    M[i][:, 0:-1:2, :],
                    einsum("nxaijk,nxkl->nxaijl", M[i][:, 1::2], T[i][:, 0:-1:2]),
                ],
                axis=2,
            )
        )
    return M


@static_jit(static_argnums=(3,))
def gating_matrices(
    S,
    T,
    M,
    top_level: int,
    einsum: Callable[[str, ...], jax.Array] = jnp.einsum,
    null_elem: Callable[[jax.Array], jax.Array] = identity,
) -> list[jax.Array]:
    """
    >>> from .util import allclose_verbose
    >>> T = transition_matrices(0.5*jnp.ones([1, 4, 1, 1]), 2)
    >>> bool(allclose_verbose(
    ...     gating_matrices(
    ...         source_matrices(1.+jnp.arange(4)[None, :, None, None, None], T, 2),
    ...         T, mark_matrices(4.-jnp.arange(4)[None, :, None, None, None], T, 2), 2)[:, 0],
    ...     jnp.array([[[[[[[0.]]]], [[[[0.]]]], [[[[0.]]]], [[[[0]]]]],
    ...                    [[[[[3.]]]], [[[[0.]]]], [[[[0.]]]], [[[[0.]]]]],
    ...                    [[[[[1.]]]], [[[[4.]]]], [[[[0.]]]], [[[[0.]]]]],
    ...                    [[[[[0.25]]]], [[[[1.]]]], [[[[3.]]]], [[[[0.]]]]]]])))
    True
    """
    Gn = []
    B, X, _, JT, JK, JV = S[0].shape
    _, _, _, JO, JQ, _ = M[0].shape
    C = 1 << top_level
    Xc = X // C

    for i in range(top_level):
        Gn.append(
            jnp.concatenate(
                [
                    jnp.zeros(
                        [
                            B,
                            Xc,
                            C >> (i + 1),
                            S[i].shape[2],
                            2 * S[i].shape[2],
                            JO,
                            JQ,
                            JK,
                            JV,
                        ],
                        dtype=S[0].dtype,
                    ),
                    jnp.concatenate(
                        [
                            einsum(
                                "nxaijk,nxbklm->nxabijlm",
                                M[i][:, 1::2, :],
                                S[i][:, 0:-1:2, :],
                            ).reshape(
                                (
                                    B,
                                    Xc,
                                    C >> (i + 1),
                                    S[i].shape[2],
                                    S[i].shape[2],
                                    JO,
                                    JQ,
                                    JK,
                                    JV,
                                )
                            ),
                            jnp.zeros(
                                [
                                    B,
                                    Xc,
                                    C >> (i + 1),
                                    S[i].shape[2],
                                    S[i].shape[2],
                                    JO,
                                    JQ,
                                    JK,
                                    JV,
                                ],
                                dtype=S[0].dtype,
                            ),
                        ],
                        axis=4,
                    ),
                ],
                axis=3,
            )
        )
    if top_level == 0:
        G = jnp.zeros((B, Xc, C, C, JO, JQ, JK, JV))
    else:
        G = sum(
            [
                (
                    g[:, :, :, :, None]
                    * (jnp.eye(g.shape[2], dtype=g.dtype)[None, None, :, None, :, None, None, None, None, None])
                ).reshape(B, Xc, C, C, JO, JQ, JK, JV)
                for g in Gn
            ]
        )
    return G


@static_jit(static_argnums=(8, 9, 10, 11, 12))
def pLSTM1D_jax(
    Q,
    K,
    V,
    S0,
    T0,
    M0,
    D0,
    C_initial=None,
    levels: int = 64,
    return_last_C=None,
    QK_scale=None,
    dtype: DTYPES = "bfloat16",
    state_dtype: DTYPES = "float32",
):
    DB, DT, DHQK, DHHV, JT, JK, JV, JO, JQ = (
        Q.shape[0],
        Q.shape[1],
        Q.shape[2],
        V.shape[2],
        S0.shape[2],
        S0.shape[3],
        S0.shape[4],
        M0.shape[2],
        M0.shape[3],
    )
    BT = 1 << levels
    NT = DT // BT
    if QK_scale is None:
        QK_scale = DHQK ** (-1 / 2)
    top_level = log2(BT)
    T = transition_matrices(T0, top_level=top_level)
    S = source_matrices(S0, T, top_level=top_level)
    M = mark_matrices(M0, T, top_level=top_level)
    D = D0.reshape(DB, NT, BT, JO, JQ, JK, JV)
    G = (
        gating_matrices(S, T, M, top_level=top_level)
        + D[:, :, :, None] * jnp.eye(BT, dtype=D.dtype)[None, None, :, :, None, None, None, None]
    )
    T, S, M, G = T[-1], S[-1], M[-1], G
    Q = Q.reshape((DB, NT, BT, DHQK, JQ))
    K = K.reshape((DB, NT, BT, DHQK, JK))
    V = V.reshape((DB, NT, BT, DHHV, JV))

    inter_chunk_Skv = jnp.einsum(
        "ncaijk,ncadj,ncavk->ncdvi",
        S,
        K,
        V,
    )
    C = jnp.zeros([DB, DHQK, DHHV, JT], dtype=Q.dtype)
    inter_chunks = []
    if C_initial is not None:
        C = C_initial
    for i in range(1, NT + 1):
        inter_chunks.append(QK_scale * jnp.einsum("nadj,naijk,ndvk->navi", Q[:, i - 1], M[:, i - 1], C))
        C = jnp.einsum("nij,ndvj->ndvi", T[:, i - 1], C) + inter_chunk_Skv[:, i - 1]

    inter_chunk = jnp.stack(inter_chunks, axis=1)
    intra_chunk = QK_scale * jnp.einsum(
        "ncadj,ncabijkl,ncbdk,ncbvl->ncavi",
        Q,
        G,
        K,
        V,
    )
    if return_last_C:
        return (inter_chunk + intra_chunk).reshape(DB, DT, DHHV, JO), C
    else:
        return (inter_chunk + intra_chunk).reshape(DB, DT, DHHV, JO)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)

    DB, DT, DHQK, DHHV, JQ, JK, JV, JT, JO = (3, 8, 32, 32, 1, 1, 1, 4, 1)

    Q, K, V = (
        jnp.ones([DB, DT, DHQK, JQ]),
        jnp.ones([DB, DT, DHQK, JK]),
        jnp.ones([DB, DT, DHHV, JV]),
    )

    S0, T0, M0, D0 = (
        jnp.ones([DB, DT, JT, JK, JV]),
        jnp.ones([DB, DT, JT, JT]),
        jnp.ones([DB, DT, JO, JQ, JT]),
        jnp.ones([DB, DT, JO, JQ, JK, JV]),
    )

    res = pLSTM1D_jax(Q, K, V, S0, T0, M0, D0, levels=3)

    print(res)
