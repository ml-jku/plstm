from collections.abc import Callable
import torch

# if torch.__version__ > "2.3"
from torch.amp import custom_fwd, custom_bwd
# from torch.cuda.amp import custom_fwd, custom_bwd

# try:
from ..util import log2
from .util import identity
# except ImportError:
#     import sys
#     import os

#     sys.path.append(os.path.join(os.path.split(os.path.abspath(__file__))), "..")
#     from util import log2
#     from torch.util import identity, rev_cumsum_off


def transition_matrices(
    T0,
    levels: int = 10,
    einsum: Callable[[str, ...], torch.Tensor] = torch.einsum,
    null_elem: Callable[[torch.Tensor], torch.Tensor] = identity,
):
    """
    >>> torch.allclose(
    ...     transition_matrices(0.5*torch.ones([8])[None, :, None, None])[0],
    ...     torch.tensor([[[[0.5000]],[[0.5000]],[[0.5000]],[[0.5000]],[[0.5000]],[[0.5000]],[[0.5000]],[[0.5000]]]]))
    True
    >>> torch.allclose(
    ...     transition_matrices(1.+torch.arange(4.)[None, :, None, None])[1],
    ...     torch.tensor([[[[2.]], [[12.]]]]))
    True
    >>> torch.allclose(
    ...     transition_matrices(0.5*torch.ones([8])[None, :, None, None])[1],
    ...     torch.tensor([[[[0.2500]],[[0.2500]],[[0.2500]],[[0.2500]]]]))
    True
    >>> torch.allclose(
    ...     transition_matrices(0.5*torch.ones([8])[None, :, None, None])[2],
    ...     torch.tensor([[[[0.0625]], [[0.0625]]]]))
    True
    >>> torch.allclose(
    ...     transition_matrices(0.5*torch.ones([8])[None, :, None, None])[3],
    ...     torch.tensor([[[[0.00390625]]]]))
    True
    >>> transition_matrices(0.5*torch.ones([1, 4, 1, 1]))
    [tensor([[[[0.5000]],
    <BLANKLINE>
             [[0.5000]],
    <BLANKLINE>
             [[0.5000]],
    <BLANKLINE>
             [[0.5000]]]]), tensor([[[[0.2500]],
    <BLANKLINE>
             [[0.2500]]]]), tensor([[[[0.0625]]]])]
    """
    T = [T0]
    for i in range(min(log2(T0.shape[1]), levels)):
        T.append(
            einsum("nxij,nxjk->nxik", T[i][:, 1::2], T[i][:, 0:-1:2]),
        )
    return T


def source_matrices(
    S0,
    T,
    levels: int = 10,
    einsum: Callable[[str, ...], torch.Tensor] = torch.einsum,
    null_elem: Callable[[torch.Tensor], torch.Tensor] = identity,
):
    """
    >>> torch.allclose(
    ...     source_matrices(
    ...         1.+torch.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*torch.ones([1, 4, 1, 1])))[0],
    ...     torch.tensor([[[[1.]]], [[[2.]]], [[[3.]]], [[[4.]]]]))
    True
    >>> torch.allclose(
    ...     source_matrices(1+torch.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*torch.ones(4)[None, :, None, None]))[1],
    ...     torch.tensor([[[[0.5000]], [[2.0000]]], [[[1.5000]], [[4.0000]]]]))
    True
    >>> torch.allclose(
    ...     source_matrices(1+torch.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*torch.ones(4)[None, :, None, None]))[2],
    ...     torch.tensor([[[[0.1250]], [[0.5000]], [[1.5000]], [[4.0000]]]]))
    True
    """
    S = [S0.unsqueeze(dim=2)]
    for i in range(min(log2(S0.shape[1]), levels)):
        S.append(
            torch.cat(
                [
                    einsum("nxij,nxajkl->nxaikl", T[i][:, 1::2], S[i][:, 0:-1:2, :]),
                    S[i][:, 1::2, :],
                ],
                dim=2,
            )
        )
    return S


def mark_matrices(
    M0,
    T,
    levels: int = 10,
    einsum: Callable[[str, ...], torch.Tensor] = torch.einsum,
    null_elem: Callable[[torch.Tensor], torch.Tensor] = identity,
):
    """
    >>> torch.allclose(
    ...     mark_matrices(
    ...         1.+torch.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*torch.ones(4)[None, :, None, None]))[0],
    ...     torch.tensor([[[[[1.]]],[[[2.]]],[[[3.]]],[[[4.]]]]]))
    True
    >>> torch.allclose(
    ...     mark_matrices(
    ...         1+torch.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*torch.ones(4)[None, :, None, None]))[1],
    ...     torch.tensor([[[[1.]], [[1.]]]
    ...                   [[[3.]], [[2.]]]]))
    True
    >>> torch.allclose(
    ...     mark_matrices(
    ...         1+torch.arange(4)[None, :, None, None, None],
    ...         transition_matrices(0.5*torch.ones(4)[None, :, None, None]))[2],
    ...     torch.tensor([[[[1.0000]], [[1.0000]], [[0.7500]], [[0.5000]]]]))
    True
    """
    M = [M0.unsqueeze(dim=2)]
    for i in range(min(log2(M0.shape[1]), levels)):
        M.append(
            torch.cat(
                [
                    M[i][:, 0:-1:2, :],
                    einsum("nxaijk,nxkl->nxaijl", M[i][:, 1::2], T[i][:, 0:-1:2]),
                ],
                dim=2,
            )
        )
    return M


def gating_matrices(
    S,
    T,
    M,
    levels: int = 10,
    einsum: Callable[[str, ...], torch.Tensor] = torch.einsum,
    null_elem: Callable[[torch.Tensor], torch.Tensor] = identity,
):
    """
    >>> T = transition_matrices(0.5*torch.ones([1, 4, 1, 1]))
    >>> torch.allclose(
    ...     gating_matrices(
    ...         source_matrices(1.+torch.arange(4)[None, :, None, None, None], T),
    ...         mark_matrices(4.-torch.arange(4)[None, :, None, None, None], T), T)[:, 0],
    ...     torch.tensor([[[[[0.]], [[0.]], [[0.]], [[0]]],
    ...                    [[[3.]], [[0.]], [[0.]], [[0.]]],
    ...                    [[[1.]], [[4.]], [[0.]], [[0.]]],
    ...                    [[[0.25]], [[1.]], [[3.]], [[0.]]]]]))
    True
    """
    Gn = []
    B, X, _, JT, JK, JV = S[0].shape
    _, _, _, JO, JQ, _ = M[0].shape
    C = min(1 << log2(X), 1 << levels)
    Xc = X // C

    for i in range(levels):
        Gn.append(
            torch.cat(
                [
                    S[0].new_zeros(
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
                        ]
                    ),
                    torch.cat(
                        [
                            einsum(
                                "nxaijk,nxbklm->nxabijlm",
                                M[i][:, 1::2, :],
                                S[i][:, 0:-1:2, :],
                            ).view(
                                B,
                                Xc,
                                C >> (i + 1),
                                S[i].shape[2],
                                S[i].shape[2],
                                JO,
                                JQ,
                                JK,
                                JV,
                            ),
                            S[0].new_zeros(
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
                                ]
                            ),
                        ],
                        dim=4,
                    ),
                ],
                dim=3,
            )
        )
    if levels == 0:
        G = S[0].new_zeros((B, Xc, C, C, JO, JQ, JK, JV))
    else:
        G = sum(
            [
                (
                    g[:, :, :, :, None]
                    * (
                        torch.eye(g.shape[2], device=g.device, dtype=g.dtype)[
                            None, None, :, None, :, None, None, None, None, None
                        ]
                    )
                ).reshape(B, Xc, C, C, JO, JQ, JK, JV)
                for g in Gn
            ]
        )
    return G


def pLSTM1D_torch(
    Q,
    K,
    V,
    S0,
    T0,
    M0,
    D0,
    C_initial=None,
    E_initial=None,
    levels: int = 6,
    return_last_C=None,
    QK_scale=None,
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
    T = transition_matrices(T0, levels=levels, einsum=torch.einsum)
    S = source_matrices(S0, T, levels=levels, einsum=torch.einsum)
    M = mark_matrices(M0, T, levels=levels, einsum=torch.einsum)
    D = D0.reshape(DB, NT, BT, JO, JQ, JK, JV)
    G = (
        gating_matrices(S, T, M, levels=levels, einsum=torch.einsum)
        + D[:, :, :, None] * torch.eye(BT, device=D.device, dtype=D.dtype)[None, None, :, :, None, None, None, None]
    )
    T, S, M, G = T[-1], S[-1], M[-1], G
    Q = Q.view(DB, NT, BT, DHQK, JQ)
    K = K.view(DB, NT, BT, DHQK, JK)
    V = V.view(DB, NT, BT, DHHV, JV)
    inter_chunk_Skv = torch.einsum(
        "ncaijk,ncadj,ncavk->ncdvi",
        S,
        K,
        V,
    )
    C = Q.new_zeros([DB, DHQK, DHHV, JT])
    inter_chunks = []
    if C_initial is not None:
        C = C_initial
    for i in range(1, NT + 1):
        inter_chunks.append(QK_scale * torch.einsum("nadj,naijk,ndvk->navi", Q[:, i - 1], M[:, i - 1], C))
        C = torch.einsum("nij,ndvj->ndvi", T[:, i - 1], C) + inter_chunk_Skv[:, i - 1]

    inter_chunk = torch.stack(inter_chunks, dim=1)
    intra_chunk = QK_scale * torch.einsum(
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


def pLSTM1D_fwbw(
    Q,
    K,
    V,
    S0,
    T0,
    M0,
    D0,
    C_initial=None,
    levels: int = 64,
    recompute_G: bool = True,
    recompute_C: bool = True,
    use_initial_C: bool = False,
    return_last_C: bool = False,
    QK_scale: int | None = None,
    einsum: Callable[[str, ...], torch.Tensor] = torch.einsum,
):
    if QK_scale is None:
        QK_scale = Q.shape[2] ** (-1 / 2)

    class pLSTM1DFunc(torch.autograd.Function):
        @staticmethod
        # @custom_fwd
        @custom_fwd(device_type="cuda" if torch.cuda.is_available() else "cpu")
        def forward(ctx, Q, K, V, S0, T0, M0, D0, C_initial=None):
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
            with torch.enable_grad():
                D = D0.reshape(DB, NT, BT, JO, JQ, JK, JV)
                T = transition_matrices(T0, levels=levels, einsum=einsum)
                S = source_matrices(S0, T, levels=levels, einsum=einsum)
                M = mark_matrices(M0, T, levels=levels, einsum=einsum)
                G = (
                    gating_matrices(S, T, M, levels=levels, einsum=einsum)
                    + D[:, :, :, None]
                    * torch.eye(BT, device=D.device, dtype=D.dtype)[None, None, :, :, None, None, None, None]
                )
                T, S, M, G = (
                    T[-1],
                    S[-1],
                    M[-1],
                    G,
                )
            Q = Q.view(DB, NT, BT, DHQK, JQ)
            K = K.view(DB, NT, BT, DHQK, JK)
            V = V.view(DB, NT, BT, DHHV, JV)

            inter_chunk_Skv = torch.einsum(
                "ncaijk,ncadj,ncavk->ncdvi",
                S,
                K,
                V,
            )
            C = Q.new_zeros([DB, NT + 1, DHQK, DHHV, JT])
            if C_initial is not None:
                C[:, 0] = C_initial
            for i in range(1, NT + 1):
                C[:, i] = (
                    torch.einsum(
                        "nij,ndvj->ndvi",
                        T[:, i - 1],
                        C[:, i - 1],
                    )
                    + inter_chunk_Skv[:, i - 1]
                )

            inter_chunk = QK_scale * torch.einsum("ncadj,ncaijk,ncdvk->ncavi", Q, M, C[:, :-1])

            intra_chunk = QK_scale * torch.einsum(
                "ncadj,ncabijkl,ncbdk,ncbvl->ncavi",
                Q,
                G,
                K,
                V,
            )

            ctx.save_for_backward(
                Q,
                K,
                V,
                S0,
                T0,
                M0,
                D0,
                C if not recompute_C else None,
                S if not recompute_G else None,
                T if not recompute_G else None,
                M if not recompute_G else None,
                G if not recompute_G else None,
                C_initial,
            )

            if return_last_C:
                return (
                    (inter_chunk + intra_chunk).reshape(DB, DT, DHHV, JO).detach(),
                    C[:, NT].detach(),
                )
            else:
                return (inter_chunk + intra_chunk).reshape(DB, DT, DHHV, JO).detach()

        @staticmethod
        # @custom_bwd
        @custom_bwd(device_type="cuda" if torch.cuda.is_available() else "cpu")
        def backward(ctx, dY, dC_last=None):
            (
                Q,
                K,
                V,
                S0,
                T0,
                M0,
                D0,
                C,
                S,
                T,
                M,
                G,
                C_initial,
            ) = ctx.saved_tensors
            DB, NT, BT, DHQK, DHHV, JT, JK, JV, JO, JQ = (
                Q.shape[0],
                Q.shape[1],
                Q.shape[2],
                Q.shape[3],
                V.shape[3],
                S0.shape[2],
                S0.shape[3],
                S0.shape[4],
                M0.shape[2],
                M0.shape[3],
            )
            DT = NT * BT
            levels = log2(BT)

            dY = dY.reshape(DB, NT, BT, DHHV, JO)

            if G is None:
                with torch.enable_grad():
                    D = D0.reshape(DB, NT, BT, JO, JQ, JK, JV)
                    T = transition_matrices(T0, levels=levels, einsum=einsum)
                    S = source_matrices(S0, T, levels=levels, einsum=einsum)
                    M = mark_matrices(M0, T, levels=levels, einsum=einsum)
                    G = (
                        gating_matrices(S, T, M, levels=levels, einsum=einsum)
                        + D[:, :, :, None]
                        * torch.eye(BT, device=D.device, dtype=D.dtype)[None, None, :, :, None, None, None, None]
                    )
                    T, S, M, G = T[-1], S[-1], M[-1], G

            if C is None:
                inter_chunk_Skv = torch.einsum(
                    "ncaijk,ncadj,ncavk->ncdvi",
                    S,
                    K,
                    V,
                )
                C = Q.new_zeros([DB, NT + 1, DHQK, DHHV, JT])
                if C_initial is not None:
                    C[:, 0] = C_initial
                for i in range(1, NT + 1):
                    C[:, i] = torch.einsum("nij,ndvj->ndvi", T[:, i - 1], C[:, i - 1]) + inter_chunk_Skv[:, i - 1]
            dC = torch.zeros_like(C)
            dT = torch.zeros_like(T)
            dS = torch.zeros_like(S)
            dM = torch.zeros_like(M)
            dG = torch.zeros_like(G)
            dQ = torch.zeros_like(Q)
            dK = torch.zeros_like(K)
            dV = torch.zeros_like(V)

            # inter chunk
            if return_last_C and dC_last is not None:
                dC[:, -1] = dC_last
                dC[:, :-1] = torch.einsum("ncadj,ncaijk,ncavi->ncdvk", QK_scale * Q, M, dY)
                for i in range(NT, 0, -1):
                    dC[:, i - 1] += torch.einsum("nij,ndvi->ndvj", T[:, i - 1], dC[:, i])
                    dT[:, i - 1] = torch.einsum("ndvj,ndvi->nij", C[:, i - 1], dC[:, i])

                dCV = torch.einsum("ncdvi,ncavk->ncadik", dC[:, 1:], V)
                dS = torch.einsum("ncadik,ncadj->ncaijk", dCV, K)
                dK = torch.einsum("ncadik,ncaijk->ncadj", dCV, S)
                dV = torch.einsum("ncdvi,ncadj,ncaijk->ncavk", dC[:, 1:], K, S)

                dCY = QK_scale * torch.einsum("ncdvj,ncavi->ncadij", C[:, :-1], dY)
                dQ = torch.einsum("ncaijk,ncadik->ncadj", M, dCY)
                dM = torch.einsum("ncadj,ncadik->ncaijk", Q, dCY)

            # intra chunk
            dYGV = torch.einsum("ncavi,ncabijkl,ncbvl->ncabjk", dY, G, V)
            dK += QK_scale * torch.einsum("ncabij,ncadi->ncbdj", dYGV, Q)
            dQ += QK_scale * torch.einsum("ncbdj,ncabij->ncadi", K, dYGV)
            dV += QK_scale * torch.einsum("ncavi,ncabijkl,ncbdk,ncadj->ncbvl", dY, G, K, Q)
            dG += QK_scale * torch.einsum("ncavi,ncadj,ncbdk,ncbvl->ncabijkl", dY, Q, K, V)

            dS0, dT0, dM0, dD0 = None, None, None, None
            dS0, dT0, dM0, dD0 = torch.autograd.grad(
                outputs=(T, S, M, G),
                inputs=(S0, T0, M0, D0),
                grad_outputs=(dT, dS, dM, dG),
                retain_graph=True,
                create_graph=True,
            )
            dQ = dQ.reshape(DB, DT, DHQK, JQ)
            dK = dK.reshape(DB, DT, DHQK, JK)
            dV = dV.reshape(DB, DT, DHHV, JV)

            if use_initial_C:
                return dQ, dK, dV, dS0, dT0, dM0, dD0, dC[:, 0]
            else:
                return dQ, dK, dV, dS0, dT0, dM0, dD0, None

    return pLSTM1DFunc.apply(Q, K, V, S0, T0, M0, D0, C_initial)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
