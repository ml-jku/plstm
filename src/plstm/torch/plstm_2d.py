from collections.abc import Callable
import torch

from torch.amp import custom_fwd, custom_bwd

# from torch.cuda.amp import custom_fwd, custom_bwd
from ..util import log2
from .util import plus, identity

"""
Naming conventions:
Indices x,y.. refer to positions in the big grid (e.g. up to X//2**level)
Indices a,b.. refer to indices along borders
Indices w,z.. refer to indices within a current (meta-)cell (e.g. up to 2**level)
Indices i,j.. refer to indices for transition internal matrices (in case of more than scalar transitions)
Indices n,m.. refer to batch size, head size etc

"""


def transition_matrices_2d(
    T0_rl,
    T0_du,
    T0_dl,
    T0_ru,
    levels: int = 10,
    einsum: Callable[[str, ...], torch.Tensor] = torch.einsum,
    acc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = plus,
    null_elem: Callable[[torch.Tensor], torch.Tensor] = identity,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    T0: base transition matrices going from one adjacent point to another adjacent one
        [N, X, Y, A, A, J, J]

    >>> torch.allclose(
    ...     transition_matrices_2d({
    ...         "rl": torch.ones([1, 2, 2, 1, 1, 1, 1]), "ru": torch.ones([1, 2, 2, 1, 1, 1, 1]),
    ...         "dl": torch.ones([1, 2, 2, 1, 1, 1, 1]), "du": torch.ones([1, 2, 2, 1, 1, 1, 1])})[1]["rl"],
    ...     torch.tensor([[[[[[[1.]], [[0.]]], [[[2.]], [[1.]]]]]]]))
    True
    """
    B, X, Y, J, _ = T0_rl.shape
    T_rl = [T0_rl[:, :, :, None, None]]
    T_du = [T0_du[:, :, :, None, None]]
    T_dl = [T0_dl[:, :, :, None, None]] if T0_dl is not None else None
    T_ru = [T0_ru[:, :, :, None, None]] if T0_ru is not None else None

    for i in range(log2(min(T0_du.shape[1], T0_du.shape[2], (1 << levels)))):
        T_rl.append(
            torch.cat(
                [
                    torch.cat(
                        [
                            einsum(
                                "nxyabij,nxybcjk->nxyacik",
                                T_rl[i][:, 1::2, 0:-1:2],
                                T_rl[i][:, 0:-1:2, 0:-1:2],
                            ),
                            null_elem(torch.zeros_like(T_rl[i][:, 1::2, 0:-1:2])),
                        ],
                        dim=4,
                    ),
                    torch.cat(
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
                                else null_elem(torch.zeros_like(T_rl[i][:, 1::2, 0:-1:2]))
                            ),
                            einsum(
                                "nxyabij,nxybcjk->nxyacik",
                                T_rl[i][:, 1::2, 1::2],
                                T_rl[i][:, 0:-1:2, 1::2],
                            ),
                        ],
                        dim=4,
                    ),
                ],
                dim=3,
            )
        )
        T_du.append(
            torch.cat(
                [
                    torch.cat(
                        [
                            einsum(
                                "nxyabij,nxybcjk->nxyacik",
                                T_du[i][:, 0:-1:2, 1::2],
                                T_du[i][:, 0:-1:2, 0:-1:2],
                            ),
                            null_elem(torch.zeros_like(T_rl[i][:, 1::2, 0:-1:2])),
                        ],
                        dim=4,
                    ),
                    torch.cat(
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
                                else null_elem(torch.zeros_like(T_rl[i][:, 1::2, 0:-1:2]))
                            ),
                            einsum(
                                "nxyabij,nxybcjk->nxyacik",
                                T_du[i][:, 1::2, 1::2],
                                T_du[i][:, 1::2, 0:-1:2],
                            ),
                        ],
                        dim=4,
                    ),
                ],
                dim=3,
            )
        )
        if T_ru is not None:
            T_ru.append(
                torch.cat(
                    [
                        torch.cat(
                            [
                                einsum(
                                    "nxyabij,nxybcjk->nxyacik",
                                    T_rl[i][:, 1::2, 0:-1:2],
                                    T_ru[i][:, 0:-1:2, 0:-1:2],
                                ),
                                T_ru[i][:, 1::2, 0:-1:2],
                            ],
                            dim=4,
                        ),
                        torch.cat(
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
                                        else null_elem(torch.zeros_like(T_du[i][:, 1::2, 1::2]))
                                    ),
                                ),
                                einsum(
                                    "nxyabij,nxybcjk->nxyacik",
                                    T_ru[i][:, 1::2, 1::2],
                                    T_du[i][:, 1::2, 0:-1:2],
                                ),
                            ],
                            dim=4,
                        ),
                    ],
                    dim=3,
                )
            )
        if T_dl is not None:
            T_dl.append(
                torch.cat(
                    [
                        torch.cat(
                            [
                                einsum(
                                    "nxyabij,nxybcjk->nxyacik",
                                    T_du[i][:, 0:-1:2, 1::2],
                                    T_dl[i][:, 0:-1:2, 0:-1:2],
                                ),
                                T_dl[i][:, 0:-1:2, 1::2],
                            ],
                            dim=4,
                        ),
                        torch.cat(
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
                                        else null_elem(torch.zeros_like(T_du[i][:, 0:-1:2, 0:-1:2]))
                                    ),
                                ),
                                einsum(
                                    "nxyabij,nxybcjk->nxyacik",
                                    T_dl[i][:, 1::2, 1::2],
                                    T_rl[i][:, 0:-1:2, 1::2],
                                ),
                            ],
                            dim=4,
                        ),
                    ],
                    dim=3,
                )
            )
    return T_rl, T_du, T_dl, T_ru


def _source_reshape(X: torch.Tensor):
    return X[:, : (X.shape[1] >> 1) << 1, :, : (X.shape[3] >> 1) << 1 :].view(
        X.shape[0],
        X.shape[1] >> 1,
        X.shape[2] << 1,
        X.shape[3] >> 1,
        X.shape[4] << 1,
        *X.shape[5:],
    )


def source_matrices_2d(
    S0_r: torch.Tensor,
    S0_d: torch.Tensor,
    T_rl: list[torch.Tensor],
    T_du: list[torch.Tensor],
    T_dl: list[torch.Tensor],
    T_ru: list[torch.Tensor],
    levels: int = 10,
    einsum: Callable[[str, ...], torch.Tensor] = torch.einsum,
    acc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = plus,
    null_elem: Callable[[torch.Tensor], torch.Tensor] = identity,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    S_r = [S0_r[:, :, None, :, None, None]]
    S_d = [S0_d[:, :, None, :, None, None]]
    for i in range(log2(min(S0_r.shape[1], S0_r.shape[2], (1 << levels)))):
        Sr = _source_reshape(S_r[i])
        Sd = _source_reshape(S_d[i])
        S_r.append(
            torch.cat(
                [
                    torch.cat(
                        [
                            torch.cat(
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
                                        else null_elem(torch.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                        einsum(
                                            "nxyabij,nxybcjk,nxwyzcklm->nxwyzailm",
                                            T_rl[i][:, 1::2, 1::2],
                                            T_ru[i][:, 0:-1:2, 1::2],
                                            Sd[:, :, : (1 << i), :, : (1 << i), :],
                                        )
                                        if T_ru is not None
                                        else null_elem(torch.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    ),
                                ],
                                dim=5,
                            ),
                            torch.cat(
                                [
                                    null_elem(torch.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    einsum(
                                        "nxyabij,nxwyzbjkl->nxwyzaikl",
                                        T_rl[i][:, 1::2, 1::2],
                                        Sr[:, :, : (1 << i), :, (1 << i) :, :],
                                    ),
                                ],
                                dim=5,
                            ),
                        ],
                        dim=4,
                    ),
                    torch.cat(
                        [
                            torch.cat(
                                [
                                    Sr[:, :, (1 << i) :, :, : (1 << i), :],
                                    (
                                        einsum(
                                            "nxyabij,nxwyzbjkl->nxwyzaikl",
                                            T_ru[i][:, 1::2, 1::2],
                                            Sd[:, :, (1 << i) :, :, : (1 << i), :],
                                        )
                                        if T_ru is not None
                                        else null_elem(torch.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)]))
                                    ),
                                ],
                                dim=5,
                            ),
                            torch.cat(
                                [
                                    null_elem(torch.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    Sr[:, :, (1 << i) :, :, (1 << i) :, :],
                                ],
                                dim=5,
                            ),
                        ],
                        dim=4,
                    ),
                ],
                dim=2,
            )
        )
        S_d.append(
            torch.cat(
                [
                    torch.cat(
                        [
                            torch.cat(
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
                                            torch.zeros_like(
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
                                        else null_elem(torch.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    ),
                                ],
                                dim=5,
                            ),
                            torch.cat(
                                [
                                    Sd[:, :, : (1 << i), :, (1 << i) :, :],
                                    (
                                        einsum(
                                            "nxyabij,nxwyzbjkl->nxwyzaikl",
                                            T_dl[i][:, 1::2, 1::2],
                                            Sr[:, :, : (1 << i), :, (1 << i) :, :],
                                        )
                                        if T_dl is not None
                                        else null_elem(torch.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)]))
                                    ),
                                ],
                                dim=5,
                            ),
                        ],
                        dim=4,
                    ),
                    torch.cat(
                        [
                            torch.cat(
                                [
                                    null_elem(torch.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    einsum(
                                        "nxyabij,nxwyzbjkl->nxwyzaikl",
                                        T_du[i][:, 1::2, 1::2],
                                        Sd[:, :, (1 << i) :, :, : (1 << i), :],
                                    ),
                                ],
                                dim=5,
                            ),
                            torch.cat(
                                [
                                    null_elem(torch.zeros_like(Sr[:, :, : (1 << i), :, : (1 << i)])),
                                    Sd[:, :, (1 << i) :, :, (1 << i) :, :],
                                ],
                                dim=5,
                            ),
                        ],
                        dim=4,
                    ),
                ],
                dim=2,
            )
        )
    return S_r, S_d


def mark_matrices_2d(
    M0_l: torch.Tensor,
    M0_u: torch.Tensor,
    T_rl: list[torch.Tensor],
    T_du: list[torch.Tensor],
    T_dl: list[torch.Tensor],
    T_ru: list[torch.Tensor],
    levels: int = 10,
    einsum: Callable[[str, ...], torch.Tensor] = torch.einsum,
    acc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = plus,
    null_elem: Callable[[torch.Tensor], torch.Tensor] = identity,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    M_l = [M0_l[:, :, None, :, None, None]]
    M_u = [M0_u[:, :, None, :, None, None]]
    for i in range(log2(min(M0_l.shape[1], M0_u.shape[2], 1 << levels))):
        Mu = _source_reshape(M_u[i])
        Ml = _source_reshape(M_l[i])
        M_l.append(
            torch.cat(
                [
                    torch.cat(
                        [
                            torch.cat(
                                [
                                    Ml[:, :, : (1 << i), :, : (1 << i)],
                                    null_elem(torch.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                ],
                                dim=5,
                            ),
                            torch.cat(
                                [
                                    (
                                        einsum(
                                            "nxwyzaijk,nxyabkl->nxwyzbijl",
                                            Mu[:, :, : (1 << i), :, (1 << i) :, :],
                                            T_dl[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_dl is not None
                                        else null_elem(torch.zeros_like(Mu[:, :, : (1 << i), :, : (1 << i)]))
                                    ),
                                    Ml[:, :, : (1 << i), :, (1 << i) :, :],
                                ],
                                dim=5,
                            ),
                        ],
                        dim=4,
                    ),
                    torch.cat(
                        [
                            torch.cat(
                                [
                                    einsum(
                                        "nxwyzaijk,nxyabkl->nxwyzbijl",
                                        Ml[:, :, (1 << i) :, :, : (1 << i), :],
                                        T_rl[i][:, 0:-1:2, 0:-1:2],
                                    ),
                                    null_elem(torch.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                ],
                                dim=5,
                            ),
                            torch.cat(
                                [
                                    acc(
                                        einsum(
                                            "nxwyzaijk,nxyabkl,nxybclm->nxwyzcijm",
                                            Ml[:, :, (1 << i) :, :, (1 << i) :, :],
                                            T_ru[i][:, 0:-1:2, 1::2],
                                            T_dl[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_dl is not None and T_ru is not None
                                        else null_elem(torch.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                        einsum(
                                            "nxwyzaijk,nxyabkl,nxybclm->nxwyzcijm",
                                            Mu[:, :, (1 << i) :, :, (1 << i) :, :],
                                            T_dl[i][:, 1::2, 0:-1:2],
                                            T_rl[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_dl is not None
                                        else null_elem(torch.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                    ),
                                    einsum(
                                        "nxwyzaijk,nxyabkl->nxwyzbijl",
                                        Ml[:, :, (1 << i) :, :, (1 << i) :, :],
                                        T_rl[i][:, 0:-1:2, 1::2],
                                    ),
                                ],
                                dim=5,
                            ),
                        ],
                        dim=4,
                    ),
                ],
                dim=2,
            )
        )
        M_u.append(
            torch.cat(
                [
                    torch.cat(
                        [
                            torch.cat(
                                [
                                    Mu[:, :, : (1 << i), :, : (1 << i)],
                                    null_elem(torch.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                ],
                                dim=5,
                            ),
                            torch.cat(
                                [
                                    einsum(
                                        "nxwyzaijk,nxyabkl->nxwyzbijl",
                                        Mu[:, :, : (1 << i), :, (1 << i) :, :],
                                        T_du[i][:, 0:-1:2, 0:-1:2],
                                    ),
                                    null_elem(torch.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)])),
                                ],
                                dim=5,
                            ),
                        ],
                        dim=4,
                    ),
                    torch.cat(
                        [
                            torch.cat(
                                [
                                    (
                                        einsum(
                                            "nxwyzaijk,nxyabkl->nxwyzbijl",
                                            Ml[:, :, (1 << i) :, :, : (1 << i), :],
                                            T_ru[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_ru is not None
                                        else null_elem(torch.zeros_like(Mu[:, :, : (1 << i), :, : (1 << i)]))
                                    ),
                                    Mu[:, :, (1 << i) :, :, : (1 << i), :],
                                ],
                                dim=5,
                            ),
                            torch.cat(
                                [
                                    (
                                        einsum(
                                            "nxwyzaijk,nxyabkl,nxybclm->nxwyzcijm",
                                            Ml[:, :, (1 << i) :, :, (1 << i) :, :],
                                            T_ru[i][:, 0:-1:2, 1::2],
                                            T_du[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_ru is not None
                                        else null_elem(torch.zeros_like(Mu[:, :, : (1 << i), :, : (1 << i)]))
                                    )
                                    + (
                                        einsum(
                                            "nxwyzaijk,nxyabkl,nxybclm->nxwyzcijm",
                                            Mu[:, :, (1 << i) :, :, (1 << i) :, :],
                                            T_dl[i][:, 1::2, 0:-1:2],
                                            T_ru[i][:, 0:-1:2, 0:-1:2],
                                        )
                                        if T_dl is not None and T_ru is not None
                                        else null_elem(torch.zeros_like(Ml[:, :, : (1 << i), :, : (1 << i)]))
                                    ),
                                    einsum(
                                        "nxwyzaijk,nxyabkl->nxwyzbijl",
                                        Mu[:, :, (1 << i) :, :, (1 << i) :, :],
                                        T_du[i][:, 1::2, 0:-1:2],
                                    ),
                                ],
                                dim=5,
                            ),
                        ],
                        dim=4,
                    ),
                ],
                dim=2,
            )
        )
    return M_l, M_u


def gating_matrices_2d(
    S_r: list[torch.Tensor],
    S_d: list[torch.Tensor],
    M_l: list[torch.Tensor],
    M_u: list[torch.Tensor],
    T_rl: list[torch.Tensor],
    T_du: list[torch.Tensor],
    T_dl: list[torch.Tensor],
    T_ru: list[torch.Tensor],
    levels: int = 10,
    einsum: Callable[[str, ...], torch.Tensor] = torch.einsum,
    acc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = plus,
    null_elem: Callable[[torch.Tensor], torch.Tensor] = identity,
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
    G_main = S_r[0].new_zeros([B, Xc, C, C, Yc, C, C, JO, JQ, JK, JV])
    # Aggregate top left part and levels in between
    for i in range(log2(C)):  # range(levels - 1):
        g = S_r[0].new_zeros(
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
            ]
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
        g[:, :, :, (1 << i) :, : (1 << i), :, :, : (1 << i), : (1 << i)] = einsum(
            "nxwyzaijk,nxuyvaklm->nxwuyzvijlm",
            M_l[i][:, 1::2, :, 0:-1:2],
            S_r[i][:, 0:-1:2, :, 0:-1:2],
        ).view(*g_part_shape)
        g[:, :, :, : (1 << i), : (1 << i), :, :, (1 << i) :, : (1 << i)] = einsum(
            "nxwyzaijk,nxuyvaklm->nxwuyzvijlm",
            M_u[i][:, 0:-1:2, :, 1::2],
            S_d[i][:, 0:-1:2, :, 0:-1:2],
        ).view(*g_part_shape)
        g[:, :, :, (1 << i) :, : (1 << i), :, :, (1 << i) :, : (1 << i)] = (
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
        ).view(*g_part_shape)
        g[:, :, :, (1 << i) :, (1 << i) :, :, :, (1 << i) :, : (1 << i)] = einsum(
            "nxwyzaijk,nxuyvaklm->nxwuyzvijlm",
            M_u[i][:, 1::2, :, 1::2],
            S_d[i][:, 1::2, :, 0:-1:2],
        ).view(*g_part_shape)
        g[:, :, :, (1 << i) :, : (1 << i), :, :, (1 << i) :, (1 << i) :] = einsum(
            "nxwyzaijk,nxuyvaklm->nxwuyzvijlm",
            M_l[i][:, 1::2, :, 1::2],
            S_r[i][:, 0:-1:2, :, 1::2],
        ).view(*g_part_shape)
        G_main[
            :,
            :,
            : g.shape[2] * g.shape[3],
            : g.shape[2] * g.shape[3],
            :,
            : g.shape[6] * g.shape[7],
            : g.shape[6] * g.shape[7],
        ] += (
            g[:, :, :, :, None, :, :, :, :, None, :, :, :, :]
            * (
                torch.eye(g.shape[2], device=g.device, dtype=g.dtype)[
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
                * torch.eye(g.shape[6], device=g.device, dtype=g.dtype)[
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

    return G_main


def stmg_matrices(
    S0_r: torch.Tensor,
    S0_d: torch.Tensor,
    T0_rl: torch.Tensor,
    T0_du: torch.Tensor,
    T0_dl: torch.Tensor,
    T0_ru: torch.Tensor,
    M0_l: torch.Tensor,
    M0_u: torch.Tensor,
    levels: int = 20,
    einsum: Callable[[str, ...], torch.Tensor] = torch.einsum,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    T_rl, T_du, T_dl, T_ru = transition_matrices_2d(T0_rl, T0_du, T0_dl, T0_ru, levels=levels, einsum=einsum)
    S_r, S_d = source_matrices_2d(S0_r, S0_d, T_rl, T_du, T_dl, T_ru, levels=levels, einsum=einsum)
    M_l, M_u = mark_matrices_2d(M0_l, M0_u, T_rl, T_du, T_dl, T_ru, levels=levels, einsum=einsum)
    G = gating_matrices_2d(S_r, S_d, M_l, M_u, T_rl, T_du, T_dl, T_ru, levels=levels, einsum=einsum)
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


# does not work with padding yet
def pLSTM2D_torch(
    Q,
    K,
    V,
    S0_r,
    S0_d,
    T0_rl,
    T0_du,
    T0_dl,
    T0_ru,
    M0_l,
    M0_u,
    D0,
    Sm=None,
    Tm=None,  # this is not used for now for a simplified stabilization
    C_initial_left=None,
    C_initial_top=None,
    levels: int = 4,
    recompute_G: bool = False,
    recompute_C: bool = False,
    use_initial_C: bool = False,
    return_last_C: bool = False,
    QK_scale: int | None = None,
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
    levels = log2(BP)
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
        * torch.eye(BP, device=D.device, dtype=D.dtype)[None, None, :, :, None, None, None, None, None, None, None]
        * torch.eye(BP, device=D.device, dtype=D.dtype)[None, None, None, None, None, :, :, None, None, None, None]
    )

    Q = Q.view(DB, NX, BP, NY, BP, DK, JQ)
    K = K.view(DB, NX, BP, NY, BP, DK, JK)
    V = V.view(DB, NX, BP, NY, BP, DV, JV)

    inter_chunk_Skv_r = torch.einsum(
        "nxuywaijk,nxuywdj,nxuywvk->nxyadvi",
        S_r,
        K,
        V,
    )
    inter_chunk_Skv_d = torch.einsum(
        "nxuywaijk,nxuywdj,nxuywvk->nxyadvi",
        S_d,
        K,
        V,
    )

    C_r = [[None for _ in range(NY + 1)] for _ in range(NX + 1)]
    C_d = [[None for _ in range(NY + 1)] for _ in range(NX + 1)]
    C_r[0][0] = Q.new_zeros([DB, BP, DK, DV, JT])
    C_d[0][0] = Q.new_zeros([DB, BP, DK, DV, JT])
    for ny in range(NY):
        C_d[0][ny + 1] = Q.new_zeros([DB, BP, DK, DV, JT])
    for nx in range(NX):
        C_r[nx + 1][0] = Q.new_zeros([DB, BP, DK, DV, JT])

    if C_initial_left is not None:
        for ny in range(NY):
            C_r[0][ny + 1] = C_initial_left[:, ny]
    else:
        for ny in range(NY):
            C_r[0][ny + 1] = Q.new_zeros([DB, BP, DK, DV, JT])
    if C_initial_top is not None:
        for nx in range(NX):
            C_d[nx + 1][0] = C_initial_top[:, nx]
    else:
        for nx in range(NX):
            C_d[nx + 1][0] = Q.new_zeros([DB, BP, DK, DV, JT])

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
                torch.einsum(
                    "nabij,nbdvj->nadvi",
                    T_rl[:, x - 1, y - 1],
                    C_r[x - 1][y],
                )
                + (
                    torch.einsum(
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
                torch.einsum(
                    "nabij,nbdvj->nadvi",
                    T_du[:, x - 1, y - 1],
                    C_d[x][y - 1],
                )
                + inter_chunk_Skv_d[:, x - 1, y - 1]
                + (
                    torch.einsum(
                        "nabij,nbdvj->nadvi",
                        T_dl[:, x - 1, y - 1],
                        C_r[x - 1][y],
                    )
                    if T_dl is not None
                    else 0
                )
            )
    C_r = torch.stack([torch.stack(Cy, dim=0) for Cy in C_r], dim=0).permute(2, 0, 1, 3, 4, 5, 6)
    C_d = torch.stack([torch.stack(Cy, dim=0) for Cy in C_d], dim=0).permute(2, 0, 1, 3, 4, 5, 6)

    inter_chunk = torch.einsum("nxwyudj,nxwyuaijk,nxyadvk->nxwyuvi", Q * QK_scale, M_l, C_r[:, :-1, 1:]) + torch.einsum(
        "nxwyudj,nxwyuaijk,nxyadvk->nxwyuvi", Q * QK_scale, M_u, C_d[:, 1:, :-1]
    )

    intra_chunk = torch.einsum(
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


def _custom_backward(inputs, outputs, grad_outputs, **kwargs):
    inputs_nonone = [inp for inp in inputs if inp is not None]
    input_grads = torch.autograd.grad(
        outputs=[out for out in outputs if out is not None],
        inputs=inputs_nonone,
        grad_outputs=[gout for out, gout in zip(outputs, grad_outputs) if out is not None],
        retain_graph=True,
        create_graph=True,
    )
    input_grads_all = []
    idx_nonone = 0
    for idx in range(len(inputs)):
        if inputs[idx] is None:
            input_grads_all.append(None)
        else:
            input_grads_all.append(input_grads[idx_nonone])
            idx_nonone += 1
    return input_grads_all


def pad_xy(pad_x, pad_y, *args):
    padded_args = []
    for arg in args:
        if arg is None:
            padded_args.append(None)
        elif pad_x > 0 or pad_y > 0:
            shape = arg.shape
            arg_res = arg.new_empty(shape[0], shape[1] + pad_x, shape[2] + pad_y, *shape[3:])
            arg_res[:, : shape[1], : shape[2]] = arg
            padded_args.append(arg_res)
        else:
            padded_args.append(arg)
    return padded_args


def unpad_xy(pad_x, pad_y, *args):
    unpadded_args = []
    for arg in args:
        if arg is None:
            unpadded_args.append(None)
        else:
            unpadded_args.append(arg[:, : arg.shape[1] - pad_x, : arg.shape[2] - pad_y])
    return unpadded_args


def pLSTM2D_fwbw(
    Q,
    K,
    V,
    S0_r,
    S0_d,
    T0_rl,
    T0_du,
    T0_dl,
    T0_ru,
    M0_l,
    M0_u,
    D0,
    Sm=None,
    Tm=None,  # this is not used for now for a simplified stabilization
    C_initial_left=None,
    C_initial_top=None,
    levels: int = 16,
    recompute_G: bool = True,
    recompute_C: bool = True,
    use_initial_C: bool = False,
    return_last_C: bool = False,
    apply_padding: bool = True,
    QK_scale: float | None = None,
):
    assert (
        apply_padding and not return_last_C and not use_initial_C
    ), "Padding only works without last state emission and without initial state"
    if QK_scale is None:
        QK_scale = Q.shape[3] ** (-1 / 2)

    class pLSTM2DFunc(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda" if torch.cuda.is_available() else "cpu")
        # @custom_fwd
        def forward(
            ctx,
            Q,
            K,
            V,
            S0_r,
            S0_d,
            T0_rl,
            T0_du,
            T0_dl,
            T0_ru,
            M0_l,
            M0_u,
            D0,
            C_initial_left=None,
            C_initial_top=None,
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
            BP = 1 << levels

            if apply_padding:
                pad_x = (BP - MX % BP) % BP
                pad_y = (BP - MY % BP) % BP
                NX = (MX + BP - 1) // BP
                NY = (MY + BP - 1) // BP
            else:
                pad_x = 0
                pad_y = 0
            (
                Q_,
                K_,
                V_,
                S0_r_,
                S0_d_,
                T0_rl_,
                T0_du_,
                T0_dl_,
                T0_ru_,
                M0_l_,
                M0_u_,
                D0_,
            ) = pad_xy(
                pad_x,
                pad_y,
                Q,
                K,
                V,
                S0_r,
                S0_d,
                T0_rl,
                T0_du,
                T0_dl,
                T0_ru,
                M0_l,
                M0_u,
                D0,
            )

            with torch.enable_grad():
                D = D0_.reshape(-1, NX, BP, 1, NY, BP, 1, JO, JQ, JK, JV)
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

                G = G_nodirect + (
                    D
                    * torch.eye(BP, device=D.device, dtype=D.dtype)[
                        None, None, :, :, None, None, None, None, None, None, None
                    ]
                    * torch.eye(BP, device=D.device, dtype=D.dtype)[
                        None, None, None, None, None, :, :, None, None, None, None
                    ]
                )

            Q = Q_.view(DB, NX, BP, NY, BP, DK, JQ)
            K = K_.view(DB, NX, BP, NY, BP, DK, JK)
            V = V_.view(DB, NX, BP, NY, BP, DV, JV)

            if NX > 1 or NY > 1 or return_last_C:
                inter_chunk_Skv_r = torch.einsum(
                    "nxuywaijk,nxuywdj,nxuywvk->nxyadvi",
                    S_r,
                    K,
                    V,
                )
                inter_chunk_Skv_d = torch.einsum(
                    "nxuywaijk,nxuywdj,nxuywvk->nxyadvi",
                    S_d,
                    K,
                    V,
                )

                C_r = Q.new_zeros([DB, NX + 1, NY + 1, BP, DK, DV, JT])
                C_d = Q.new_zeros([DB, NX + 1, NY + 1, BP, DK, DV, JT])

                if C_initial_left is not None:
                    C_r[:, 0, 1:] = C_initial_left.reshape([DB, NY, BP, DK, DV, JT])
                if C_initial_top is not None:
                    C_d[:, 1:, 0] = C_initial_top.reshape([DB, NX, BP, DK, DV, JT])

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
                        C_r[:, x, y] = (
                            torch.einsum(
                                "nabij,nbdvj->nadvi",
                                T_rl[:, x - 1, y - 1],
                                C_r[:, x - 1, y],
                            )
                            + inter_chunk_Skv_r[:, x - 1, y - 1]
                            + (
                                torch.einsum(
                                    "nabij,nbdvj->nadvi",
                                    T_ru[:, x - 1, y - 1],
                                    C_d[:, x, y - 1],
                                )
                                if T_ru is not None
                                else 0
                            )
                        )
                        C_d[:, x, y] = (
                            torch.einsum(
                                "nabij,nbdvj->nadvi",
                                T_du[:, x - 1, y - 1],
                                C_d[:, x, y - 1],
                            )
                            + inter_chunk_Skv_d[:, x - 1, y - 1]
                            + (
                                torch.einsum(
                                    "nabij,nbdvj->nadvi",
                                    T_dl[:, x - 1, y - 1],
                                    C_r[:, x - 1, y],
                                )
                                if T_dl is not None
                                else 0
                            )
                        )
                inter_chunk = torch.einsum(
                    "nxwyudj,nxwyuaijk,nxyadvk->nxwyuvi",
                    QK_scale * Q,
                    M_l,
                    C_r[:, :-1, 1:],
                ) + torch.einsum(
                    "nxwyudj,nxwyuaijk,nxyadvk->nxwyuvi",
                    QK_scale * Q,
                    M_u,
                    C_d[:, 1:, :-1],
                )
            else:
                inter_chunk = 0.0

            intra_chunk = torch.einsum(
                "nxwyzdj,nxwuyztijkl,nxuytdk,nxuytvl->nxwyzvi",
                QK_scale * Q,
                G,
                K,
                V,
            )
            ctx.save_for_backward(
                Q,
                K,
                V,
                S0_r,
                S0_d,
                T0_rl,
                T0_du,
                T0_dl,
                T0_ru,
                M0_l,
                M0_u,
                D0,
                Sm,
                C_r if not recompute_C else None,
                C_d if not recompute_C else None,
                S_r if not recompute_G else None,
                S_d if not recompute_G else None,
                T_rl if not recompute_G else None,
                T_du if not recompute_G else None,
                T_dl if not recompute_G else None,
                T_ru if not recompute_G else None,
                M_l if not recompute_G else None,
                M_u if not recompute_G else None,
                G if not recompute_G else None,
                C_initial_left,
                C_initial_top,
            )

            res = (inter_chunk + intra_chunk).reshape(DB, NX * BP, NY * BP, DV, JO)

            (res,) = unpad_xy(pad_x, pad_y, res)

            if return_last_C:
                return (
                    res.reshape(DB, MX, MY, DV, JO).detach(),
                    C_r[:, -1, :],
                    C_d[:, :, -1],
                )
            else:
                return res.reshape(DB, MX, MY, DV, JO).detach()

        @staticmethod
        @custom_bwd(device_type="cuda" if torch.cuda.is_available() else "cpu")
        # @custom_bwd
        def backward(ctx, dY, dC_r_last=None, dC_d_last=None):
            (
                Q,
                K,
                V,
                S0_r,
                S0_d,
                T0_rl,
                T0_du,
                T0_dl,
                T0_ru,
                M0_l,
                M0_u,
                D0,
                Sm,
                C_r,
                C_d,
                S_r,
                S_d,
                T_rl,
                T_du,
                T_dl,
                T_ru,
                M_l,
                M_u,
                G,
                C_initial_left,
                C_initial_top,
            ) = ctx.saved_tensors
            DB, NX, BP, NY, DK, DV, JT, JK, JV, JO, JQ = (
                Q.shape[0],
                Q.shape[1],
                Q.shape[2],
                Q.shape[3],
                Q.shape[5],
                V.shape[5],
                S0_r.shape[3],
                S0_r.shape[4],
                S0_r.shape[5],
                M0_l.shape[3],
                M0_l.shape[4],
            )
            MX, MY = S0_r.shape[1], S0_r.shape[2]
            pad_x = (BP - MX % BP) % BP
            pad_y = (BP - MY % BP) % BP

            (
                dY_,
                S0_r_,
                S0_d_,
                T0_rl_,
                T0_du_,
                T0_dl_,
                T0_ru_,
                M0_l_,
                M0_u_,
                D0_,
            ) = pad_xy(
                pad_x,
                pad_y,
                dY,
                S0_r,
                S0_d,
                T0_rl,
                T0_du,
                T0_dl,
                T0_ru,
                M0_l,
                M0_u,
                D0,
            )

            levels = log2(BP)
            dY = dY.reshape(DB, NX, BP, NY, BP, DV, JO)

            if G is None:
                with torch.enable_grad():
                    D = D0_.reshape(-1, NX, BP, 1, NY, BP, 1, JO, JQ, JK, JV)
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

                    G = G_nodirect + (
                        D
                        * torch.eye(BP, device=D.device, dtype=D.dtype)[
                            None, None, :, :, None, None, None, None, None, None, None
                        ]
                        * torch.eye(BP, device=D.device, dtype=D.dtype)[
                            None, None, None, None, None, :, :, None, None, None, None
                        ]
                    )

            if (C_r is None or C_d is None) and (NX > 1 or NY > 1):
                inter_chunk_Skv_r = torch.einsum(
                    "nxuywaijk,nxuywdj,nxuywvk->nxyadvi",
                    S_r,
                    K,
                    V,
                )
                inter_chunk_Skv_d = torch.einsum(
                    "nxuywaijk,nxuywdj,nxuywvk->nxyadvi",
                    S_d,
                    K,
                    V,
                )
                C_r = Q.new_zeros([DB, NX + 1, NY + 1, BP, DK, DV, JT])
                C_d = Q.new_zeros([DB, NX + 1, NY + 1, BP, DK, DV, JT])

                if C_initial_left is not None:
                    C_r[:, 0] = C_initial_left
                if C_initial_top is not None:
                    C_d[:, :, 0] = C_initial_top

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
                        C_r[:, x, y] = (
                            torch.einsum(
                                "nabij,nbdvj->nadvi",
                                T_rl[:, x - 1, y - 1],
                                C_r[:, x - 1, y],
                            )
                            + inter_chunk_Skv_r[:, x - 1, y - 1]
                            + (
                                torch.einsum(
                                    "nabij,nbdvj->nadvi",
                                    T_ru[:, x - 1, y - 1],
                                    C_d[:, x, y - 1],
                                )
                                if T_ru is not None
                                else 0
                            )
                        )
                        C_d[:, x, y] = (
                            torch.einsum(
                                "nabij,nbdvj->nadvi",
                                T_du[:, x - 1, y - 1],
                                C_d[:, x, y - 1],
                            )
                            + inter_chunk_Skv_d[:, x - 1, y - 1]
                            + (
                                torch.einsum(
                                    "nabij,nbdvj->nadvi",
                                    T_dl[:, x - 1, y - 1],
                                    C_r[:, x - 1, y],
                                )
                                if T_dl is not None
                                else 0
                            )
                        )

            dT_rl = torch.zeros_like(T_rl)
            dT_du = torch.zeros_like(T_du)
            dT_dl = torch.zeros_like(T_dl) if T_dl is not None else None
            dT_ru = torch.zeros_like(T_ru) if T_ru is not None else None

            # inter chunk
            if NX > 1 or NY > 1 or return_last_C:
                dC_r = torch.zeros_like(C_r)
                dC_d = torch.zeros_like(C_d)

                if return_last_C and dC_r_last is not None:
                    dC_r[:, -1, 1:] = dC_r_last
                else:
                    dC_r[:, -1] = 0.0
                if return_last_C and dC_d_last is not None:
                    dC_d[:, 1:, -1] = dC_d_last
                else:
                    dC_d[:, :, -1] = 0.0

                dC_r[:, :-1, 1:] = torch.einsum("nxwyudj,nxwyuaijk,nxwyuvi->nxyadvk", QK_scale * Q, M_l, dY)
                dC_d[:, 1:, :-1] = torch.einsum("nxwyudj,nxwyuaijk,nxwyuvi->nxyadvk", QK_scale * Q, M_u, dY)

                for j in range(NX + NY - 1, -1, -1):
                    for i in range(min(j + 1, NY, NX, NX + NY - j - 1)):
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

                        dC_r[:, x - 1, y] += torch.einsum(
                            "nadvi,nabij->nbdvj",
                            dC_r[:, x, y],
                            T_rl[:, x - 1, y - 1],
                        )
                        dC_d[:, x, y - 1] += torch.einsum(
                            "nadvi,nabij->nbdvj",
                            dC_d[:, x, y],
                            T_du[:, x - 1, y - 1],
                        )
                        if T_dl is not None:
                            dC_r[:, x - 1, y] += torch.einsum(
                                "nadvi,nabij->nbdvj",
                                dC_d[:, x, y],
                                T_dl[:, x - 1, y - 1],
                            )
                        if T_ru is not None:
                            dC_d[:, x, y - 1] += torch.einsum(
                                "nadvi,nabij->nbdvj",
                                dC_r[:, x, y],
                                T_ru[:, x - 1, y - 1],
                            )

                        dT_rl[:, x - 1, y - 1] = torch.einsum("nadvi,nbdvj->nabij", dC_r[:, x, y], C_r[:, x - 1, y])
                        dT_du[:, x - 1, y - 1] = torch.einsum("nadvi,nbdvj->nabij", dC_d[:, x, y], C_d[:, x, y - 1])
                        if T_dl is not None:
                            dT_dl[:, x - 1, y - 1] = torch.einsum(
                                "nadvi,nbdvj->nabij",
                                dC_d[:, x, y],
                                C_r[:, x - 1, y],
                            )
                        if T_ru is not None:
                            dT_ru[:, x - 1, y - 1] = torch.einsum(
                                "nadvi,nbdvj->nabij",
                                dC_r[:, x, y],
                                C_d[:, x, y - 1],
                            )

                    dCV_r = torch.einsum("nxyadvi,nxwyzvk->nxwyzadik", dC_r[:, 1:, 1:], V)
                    dS_r = torch.einsum("nxwyzadik,nxwyzdj->nxwyzaijk", dCV_r, K)
                    dK = torch.einsum("nxwyzadik,nxwyzaijk->nxwyzdj", dCV_r, S_r)
                    del dCV_r
                    dCV_d = torch.einsum("nxyadvi,nxwyzvk->nxwyzadik", dC_d[:, 1:, 1:], V)
                    dS_d = torch.einsum("nxwyzadik,nxwyzdj->nxwyzaijk", dCV_d, K)
                    dK += torch.einsum("nxwyzadik,nxwyzaijk->nxwyzdj", dCV_d, S_d)
                    del dCV_d
                    dV = torch.einsum("nxyadvi,nxwyzdj,nxwyzaijk->nxwyzvk", dC_r[:, 1:, 1:], K, S_r) + torch.einsum(
                        "nxyadvi,nxwyzdj,nxwyzaijk->nxwyzvk", dC_d[:, 1:, 1:], K, S_d
                    )

                    dCY_r = torch.einsum("nxwyzvi,nxyadvk->nxwyzadvik", QK_scale * dY, C_r[:, :-1, 1:])
                    dQ = torch.einsum("nxwyzadvik,nxwyzaijk->nxwyzdj", dCY_r, M_l)
                    dM_l = torch.einsum("nxwyzadvik,nxwyzdj->nxwyzaijk", dCY_r, Q)
                    del dCY_r
                    dCY_d = torch.einsum("nxwyzvi,nxyadvk->nxwyzadvik", QK_scale * dY, C_d[:, 1:, :-1])

                    dQ += torch.einsum(
                        "nxwyzadvik,nxwyzaijk->nxwyzdj",
                        dCY_d,
                        M_u,
                    )
                    dM_u = torch.einsum("nxwyzadvik,nxwyzdj->nxwyzaijk", dCY_d, Q)
                    del dCY_d
            else:
                dK = 0
                dQ = 0
                dV = 0
                dC_r = None
                dC_d = None
                dS_r = None

            # intra chunk
            dYGV = torch.einsum("nxwyzvi,nxwuyztijkl,nxuytvl->nxwuyztjk", dY, G, V)
            dK += torch.einsum("nxwuyztjk,nxwyzdj->nxuytdk", dYGV, QK_scale * Q)
            dQ += torch.einsum("nxwuyztjk,nxuytdk->nxwyzdj", dYGV, QK_scale * K)
            del dYGV
            dV += torch.einsum(
                "nxwyuvi,nxwyudj,nxwzyutijkl,nxzytdk->nxzytvl",
                dY,
                Q,
                G,
                QK_scale * K,
            )
            dG = torch.einsum(
                "nxwyuvi,nxwyudj,nxzytdk,nxzytvl->nxwzyutijkl",
                dY,
                Q,
                QK_scale * K,
                V,
            )

            if dS_r is None:
                dS0_r, dS0_d, dM0_l, dM0_u, dT0_rl, dT0_du, dT0_dl, dT0_ru, dD0 = _custom_backward(
                    outputs=(G,),
                    inputs=(S0_r, S0_d, M0_l, M0_u, T0_rl, T0_du, T0_dl, T0_ru, D0),
                    grad_outputs=(dG,),
                    retain_graph=True,
                    create_graph=True,
                )
            else:
                dS0_r, dS0_d, dM0_l, dM0_u, dT0_rl, dT0_du, dT0_dl, dT0_ru, dD0 = _custom_backward(
                    outputs=(S_r, S_d, T_rl, T_du, T_dl, T_ru, M_l, M_u, G),
                    inputs=(S0_r, S0_d, M0_l, M0_u, T0_rl, T0_du, T0_dl, T0_ru, D0),
                    grad_outputs=(
                        dS_r,
                        dS_d,
                        dT_rl,
                        dT_du,
                        dT_dl,
                        dT_ru,
                        dM_l,
                        dM_u,
                        dG,
                    ),
                    retain_graph=True,
                    create_graph=True,
                )
            (
                dQ,
                dK,
                dV,
                dS0_r,
                dS0_d,
                dT0_rl,
                dT0_du,
                dT0_dl,
                dT0_ru,
                dM0_l,
                dM0_u,
            ) = unpad_xy(
                pad_x,
                pad_y,
                dQ,
                dK,
                dV,
                dS0_r,
                dS0_d,
                dT0_rl,
                dT0_du,
                dT0_dl,
                dT0_ru,
                dM0_l,
                dM0_u,
            )

            dQ = dQ.reshape(DB, MX, MY, DK, JQ)
            dK = dK.reshape(DB, MX, MY, DK, JK)
            dV = dV.reshape(DB, MX, MY, DV, JV)

            if use_initial_C:
                return (
                    dQ,
                    dK,
                    dV,
                    dS0_r,
                    dS0_d,
                    dT0_rl,
                    dT0_du,
                    dT0_dl,
                    dT0_ru,
                    dM0_l,
                    dM0_u,
                    dD0,
                    dC_r,
                    dC_d,
                )
            else:
                return (
                    dQ,
                    dK,
                    dV,
                    dS0_r,
                    dS0_d,
                    dT0_rl,
                    dT0_du,
                    dT0_dl,
                    dT0_ru,
                    dM0_l,
                    dM0_u,
                    dD0,
                    None,
                    None,
                )

    return pLSTM2DFunc.apply(
        Q,
        K,
        V,
        S0_r,
        S0_d,
        T0_rl,
        T0_du,
        T0_dl,
        T0_ru,
        M0_l,
        M0_u,
        D0,
        C_initial_left,
        C_initial_top,
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
