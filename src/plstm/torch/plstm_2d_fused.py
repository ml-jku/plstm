import torch

from .plstm_2d import stmg_matrices


def pLSTM2D_parallel_fused_torch(
    # non-fused wrt orientation
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    # fused versions with all orientations rd, ld, ru, lu
    S0_r: torch.Tensor,  # [B, O, MX, MY, JT, JK, JV]
    S0_d: torch.Tensor,  # [B, O, MX, MY, JT, JK, JV]
    T0_rl: torch.Tensor,  # [B, O, MX, MY, JT, JT]
    T0_du: torch.Tensor,  # [B, O, MX, MY, JT, JT]
    T0_dl: torch.Tensor,  # [B, O, MX, MY, JT, JT]
    T0_ru: torch.Tensor,  # [B, O, MX, MY, JT, JT]
    M0_l: torch.Tensor,  # [B, O, MX, MY, JO, JQ, JT]
    M0_u: torch.Tensor,  # [B, O, MX, MY, JO, JQ, JT]
    # non-fused wrt orientation
    D0: torch.Tensor,
    levels: int = 16,
    recompute_G: bool = False,
    return_G: bool = False,
    QK_scale: float = None,
):
    """PyTorch implementation of pLSTM2D_parallel_fused_jax from the NNX
    implementation. This function handles the fused version of pLSTM2D with
    multiple orientations.

    Args:
        Q: Query tensor [B, MX, MY, DK, JQ]
        K: Key tensor [B, MX, MY, DK, JK]
        V: Value tensor [B, MX, MY, DV, JV]
        S0_r: Source tensor for right direction [B, O, MX, MY, JT, JK, JV]
        S0_d: Source tensor for down direction [B, O, MX, MY, JT, JK, JV]
        T0_rl: Transition tensor for right-left [B, O, MX, MY, JT, JT]
        T0_du: Transition tensor for down-up [B, O, MX, MY, JT, JT]
        T0_dl: Transition tensor for down-left [B, O, MX, MY, JT, JT]
        T0_ru: Transition tensor for right-up [B, O, MX, MY, JT, JT]
        M0_l: Mark tensor for left [B, O, MX, MY, JO, JQ, JT]
        M0_u: Mark tensor for up [B, O, MX, MY, JO, JQ, JT]
        D0: Direct tensor [B, MX, MY, JO, JQ, JK, JV]
        levels: Number of levels for the hierarchical computation
        recompute_G: Whether to recompute G
        return_G: Whether to return G
        QK_scale: Scale factor for Q*K

    Returns:
        Output tensor [B, MX, MY, DV, JO] or tuple of (output, G) if return_G is True
    """
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
    assert NX == 1, "NX must be 1 for the fused implementation"
    assert NY == 1, "NY must be 1 for the fused implementation"
    D = D0.reshape(-1, BP, 1, BP, 1, JO, JQ, JK, JV)

    def flips(A: torch.Tensor) -> torch.Tensor:
        """Apply flips to the tensor for different orientations."""
        if A is None:
            return None

        # Create a tensor with 4 orientations: original, flip X, flip Y, flip both X and Y
        return torch.stack(
            [
                A[:, 0],
                torch.flip(A[:, 1], dims=[1]),
                torch.flip(A[:, 2], dims=[2]),
                torch.flip(torch.flip(A[:, 3], dims=[1]), dims=[2]),
            ],
            dim=1,
        )

    # Apply flips to the input tensors
    S0_r_, S0_d_ = (
        flips(S0_r).reshape(DB * 4, MX, MY, JT, JK, JV),
        flips(S0_d).reshape(DB * 4, MX, MY, JT, JK, JV),
    )
    T0_rl_, T0_du_, T0_dl_, T0_ru_ = (
        flips(T0_rl).reshape(DB * 4, MX, MY, JT, JT),
        flips(T0_du).reshape(DB * 4, MX, MY, JT, JT),
        flips(T0_dl).reshape(DB * 4, MX, MY, JT, JT) if T0_dl is not None else None,
        flips(T0_ru).reshape(DB * 4, MX, MY, JT, JT) if T0_ru is not None else None,
    )
    M0_l_, M0_u_ = (
        flips(M0_l).reshape(DB * 4, MX, MY, JO, JQ, JT),
        flips(M0_u).reshape(DB * 4, MX, MY, JO, JQ, JT),
    )

    # Compute the transition, source, mark, and gating matrices
    S_r, S_d, T_rl, T_du, T_dl, T_ru, M_l, M_u, G_nodirect = stmg_matrices(
        S0_r_,
        S0_d_,
        T0_rl_,
        T0_du_,
        T0_dl_,
        T0_ru_,
        M0_l_,
        M0_u_,
        levels=levels,
    )

    # Reshape G_nodirect to handle orientations
    G_nodirect_unflipped = G_nodirect.reshape(DB, 4, MX, MX, MY, MY, JO, JQ, JK, JV)

    # Sum up all four orientations
    G_nodirect_allorientations = (
        G_nodirect_unflipped[:, 0]
        + torch.flip(torch.flip(G_nodirect_unflipped[:, 1], dims=[1]), dims=[2])
        + torch.flip(torch.flip(G_nodirect_unflipped[:, 2], dims=[3]), dims=[4])
        + torch.flip(
            torch.flip(torch.flip(torch.flip(G_nodirect_unflipped[:, 3], dims=[1]), dims=[2]), dims=[3]), dims=[4]
        )
    )

    # Add direct connections
    G = G_nodirect_allorientations + (
        D
        * torch.eye(BP, device=D.device, dtype=D.dtype)[None, :, :, None, None, None, None, None, None]
        * torch.eye(BP, device=D.device, dtype=D.dtype)[None, None, None, :, :, None, None, None, None]
    )

    # Reshape inputs for computation
    Q = Q.reshape(DB, UX, UY, DK, JQ)
    K = K.reshape(DB, UX, UY, DK, JK)
    V = V.reshape(DB, UX, UY, DV, JV)

    # Compute the intra-chunk attention
    intra_chunk = torch.einsum(
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
