import torch
from torch.nn import functional as F
from ..util import log2
from ..config.longrange_transition_layer import (
    LongRangeTransitionLayerConfig,
)
from .initialization import InitInterface


def matrix_eigval_limit(
    lmat,
    proper_grad: bool = False,
    iterations=8,
    eigval_minlimit=1.0,
    eps_norm=1e-3,
    eps_mat=1e-3,
):
    """
    >>> float(torch.linalg.eigvals(matrix_eigval_limit(torch.tensor([[[0.95, 0.1], [0.01, 1.]]]))).abs().max(dim=-1)[0])
    1.0
    """
    if proper_grad:
        mat = lmat.reshape(-1, lmat.shape[-2], lmat.shape[-1])
        B, X, Y = mat.shape
        assert X == Y, "Matrix shape has to be square"
        mat_eye = mat - torch.eye(X, device=mat.device, dtype=mat.dtype)[None, :, :]
        x = 1.0 + mat.new_zeros([B, X, 1]) + torch.arange(X, device=mat.device, dtype=mat.dtype)[None, :, None]
        for i in range(iterations):
            y = mat_eye @ x
            x = y / (torch.linalg.norm(y, dim=1, keepdim=True) + eps_norm)
        eigval = x.transpose(-2, -1) @ mat_eye @ x
        old_eigval = torch.where(eigval < 0, 1.0 + eigval, 1.0)
        mat_eye2 = mat - old_eigval * torch.eye(X, device=mat.device, dtype=mat.dtype)[None, :, :]
        for i in range(iterations):
            y = mat_eye2 @ x
            x = y / (torch.linalg.norm(y, dim=1, keepdim=True) + eps_norm)
        eigval = torch.maximum(
            x.transpose(-2, -1) @ mat_eye2 @ x + old_eigval,
            eigval_minlimit + torch.zeros_like(old_eigval),
        )

        lmat_eye = mat_eye.reshape(*lmat.shape)
        mask = (lmat_eye).abs().sum(dim=(-2, -1))[..., None, None] < eps_mat

        return torch.where(
            mask,
            lmat,
            lmat / eigval.reshape(*lmat.shape[:-2])[..., None, None],
        )
    else:

        class LimitMatrixIteration(torch.autograd.Function):
            @staticmethod
            def forward(ctx, lmat: torch.Tensor):
                mat = lmat.reshape(-1, lmat.shape[-2], lmat.shape[-1])
                B, X, Y = mat.shape
                assert X == Y, "Matrix shape has to be square"
                mat_eye = mat - torch.eye(X, device=mat.device, dtype=mat.dtype)[None, :, :]
                x = 1.0 + mat.new_zeros([B, X, 1])
                for i in range(iterations):
                    y = mat_eye @ x
                    x = y / (torch.linalg.norm(y, dim=1, keepdim=True) + eps_norm)
                eigval = x.transpose(-2, -1) @ mat_eye @ x
                old_eigval = torch.where(eigval < 0, 1.0 + eigval, 1.0)
                mat_eye2 = mat - old_eigval * torch.eye(X, device=mat.device, dtype=mat.dtype)[None, :, :]
                for i in range(iterations):
                    y = mat_eye2 @ x
                    x = y / (torch.linalg.norm(y, dim=1, keepdim=True) + eps_norm)
                eigval = torch.maximum(
                    x.transpose(-2, -1) @ mat_eye2 @ x + old_eigval,
                    eigval_minlimit + torch.zeros_like(eigval),
                )

                lmat_eye = mat_eye.reshape(*lmat.shape)
                mask = (lmat_eye).abs().sum(dim=(-2, -1))[..., None, None] < eps_mat

                ctx.save_for_backward(eigval, mask, x)

                return torch.where(
                    mask,
                    lmat,
                    lmat / eigval.reshape(*lmat.shape[:-2])[..., None, None],
                )

            # this is not the true gradient, assumes the largest eigenvector is constant
            @staticmethod
            def backward(ctx, dmat: torch.Tensor):
                (eigval, mask, x) = ctx.saved_tensors
                return torch.where(
                    mask,
                    dmat,
                    dmat / eigval.reshape(*dmat.shape[:-2])[..., None, None]
                    - (
                        (dmat.reshape(-1, dmat.shape[-2], dmat.shape[-1]) @ x) @ x.transpose(-2, -1) / eigval**2
                    ).reshape(dmat.shape),
                )

        return LimitMatrixIteration.apply(lmat)


def matrix_singval_limit(
    lmat,
    proper_grad: bool = False,
    iterations=8,
    eigval_minlimit=1.0,
    eps_norm=1e-3,
    eps_mat=1e-3,
):
    """
    >>> float(torch.linalg.eigvals(matrix_eigval_limit(torch.tensor([[[0.95, 0.1], [0.01, 1.]]]))).abs().max(dim=-1)[0])
    1.0
    """
    if proper_grad:
        mat = lmat.reshape(-1, lmat.shape[-2], lmat.shape[-1])
        B, X, Y = mat.shape
        m2 = mat @ mat.transpose(-2, -1)
        assert X == Y, "Matrix shape has to be square"
        mat_eye = m2 - torch.eye(X, device=mat.device, dtype=mat.dtype)[None, :, :]
        x = 1.0 + m2.new_zeros([B, X, 1]) + torch.arange(X, device=mat.device, dtype=mat.dtype)[None, :, None]
        for i in range(iterations):
            y = mat_eye @ x
            x = y / (torch.linalg.norm(y, dim=1, keepdim=True) + eps_norm)
        eigval = x.transpose(-2, -1) @ mat_eye @ x
        old_eigval = torch.where(eigval < 0, 1.0 + eigval, 1.0)
        mat_eye2 = m2 - old_eigval * torch.eye(X, device=mat.device, dtype=mat.dtype)[None, :, :]
        for i in range(iterations):
            y = mat_eye2 @ x
            x = y / (torch.linalg.norm(y, dim=1, keepdim=True) + eps_norm)
        eigval = torch.maximum(
            x.transpose(-2, -1) @ mat_eye2 @ x + old_eigval,
            eigval_minlimit + torch.zeros_like(old_eigval),
        )

        lmat_eye = mat_eye.reshape(*lmat.shape)
        mask = (lmat_eye).abs().sum(dim=(-2, -1))[..., None, None] < eps_mat

        singval = eigval ** (1 / 2)

        return torch.where(
            mask,
            lmat,
            lmat / singval.reshape(*lmat.shape[:-2])[..., None, None],
        )
    else:

        class LimitMatrixIteration(torch.autograd.Function):
            @staticmethod
            def forward(ctx, lmat: torch.Tensor):
                mat = lmat.reshape(-1, lmat.shape[-2], lmat.shape[-1])
                B, X, Y = mat.shape
                assert X == Y, "Matrix shape has to be square"
                m2 = mat @ mat.transpose(-2, -1)
                mat_eye = m2 - torch.eye(X, device=mat.device, dtype=mat.dtype)[None, :, :]
                x = 1.0 + m2.new_zeros([B, X, 1])
                for i in range(iterations):
                    y = mat_eye @ x
                    x = y / (torch.linalg.norm(y, dim=1, keepdim=True) + eps_norm)
                eigval = x.transpose(-2, -1) @ mat_eye @ x
                old_eigval = torch.where(eigval < 0, 1.0 + eigval, 1.0)
                mat_eye2 = m2 - old_eigval * torch.eye(X, device=mat.device, dtype=mat.dtype)[None, :, :]
                for i in range(iterations):
                    y = mat_eye2 @ x
                    x = y / (torch.linalg.norm(y, dim=1, keepdim=True) + eps_norm)
                eigval = torch.maximum(
                    x.transpose(-2, -1) @ mat_eye2 @ x + old_eigval,
                    eigval_minlimit + torch.zeros_like(eigval),
                )

                lmat_eye = mat_eye.reshape(*lmat.shape)
                mask = (lmat_eye).abs().sum(dim=(-2, -1))[..., None, None] < eps_mat

                singval = eigval ** (1 / 2)
                ctx.save_for_backward(singval, mask, x)

                return torch.where(
                    mask,
                    lmat,
                    lmat / singval.reshape(*lmat.shape[:-2])[..., None, None],
                )

            # this is not the true gradient, assumes the largest eigenvector is constant
            @staticmethod
            def backward(ctx, dmat: torch.Tensor):
                (singval, mask, x) = ctx.saved_tensors
                return torch.where(
                    mask,
                    dmat,
                    dmat / singval.reshape(*dmat.shape[:-2])[..., None, None]
                    - (
                        (dmat.reshape(-1, dmat.shape[-2], dmat.shape[-1]) @ x) @ x.transpose(-2, -1) / singval**2
                    ).reshape(dmat.shape),
                )

        return LimitMatrixIteration.apply(lmat)


def matrix_orthogonalize_householder(lmat: torch.Tensor, eps_norm: float = 1e-5):
    """Computes an orthogonal matrix from vectors via the householder
    reflections."""
    mat = lmat.reshape(-1, lmat.shape[-2], lmat.shape[-1])
    B, X, Y = mat.shape
    mat = mat / (torch.linalg.norm(mat, dim=-1, keepdim=True) + eps_norm)
    hhmat = torch.eye(X, dtype=mat.dtype, device=mat.device)[None, None, :, :] - 2 * torch.einsum(
        "nab,nac->nabc", mat, mat
    )
    hhmats = [hhmat]
    for i in range(log2(X)):
        hhmats.append(torch.einsum("nabc,nacd->nabd", hhmats[i][:, :-1:2], hhmats[i][:, 1::2]))
    return -hhmats[-1].view(lmat.shape)


def exponential_power_series(x: torch.Tensor, order: int = 8):
    orange = torch.arange(order, dtype=x.dtype, device=x.device)
    lognorm = torch.cumsum(torch.where(orange > 0, torch.log(orange), 0.0), dim=0)
    return torch.exp(torch.log(x.view(1, -1)) * orange[:, None] - lognorm[:, None]).view(order, *x.shape)


def matrix_orthogonalize_exponential(
    lmat: torch.Tensor,
    factor=0.1,
    order: int = 16,
    eps: float = 1e-3,
    # derivative_order: int = ,
    use_autograd: bool = True,
):
    """
    Calculates the approximation: exp(A) = 1 + A + 1/2*A**2 + 1/6*A**2
    """

    def _exponential_orthogonalization_autograd(lmat: torch.Tensor):
        mat = lmat.reshape(-1, lmat.shape[-2], lmat.shape[-1])
        B, X, Y = mat.shape
        assert X == Y, "Matrix has to be square"
        mat_skew = factor * (mat - mat.transpose(-2, -1))
        mat_max = torch.sum(mat_skew**2 + eps, dim=(-2, -1), keepdim=True) ** (1 / 2)  # + eps
        normalizer = torch.maximum(mat_max, torch.ones_like(mat_max))
        mat_skew_norm = mat_skew / normalizer
        identity_mat = torch.eye(X, dtype=lmat.dtype, device=lmat.device)[None, :, :] + 0.0 * mat_skew
        if order == 1:
            return identity_mat.view(lmat.shape)
        mat_skew_norm_powers2 = [mat_skew_norm]
        for i in range(log2(order - 1)):
            mat_skew_norm_powers2.append(mat_skew_norm_powers2[i] @ mat_skew_norm_powers2[i])
        mat_skew_norm_powers = [identity_mat[None, :]]
        for i in range(log2(order - 1) + 1):
            mat_skew_norm_powers.append(
                torch.cat(
                    [
                        mat_skew_norm_powers[i],
                        mat_skew_norm_powers[i] @ mat_skew_norm_powers2[i][None, :],
                    ],
                    dim=0,
                )
            )
        mat_skew_exp = torch.einsum(
            "x...,x...->...",
            exponential_power_series(normalizer, order),
            mat_skew_norm_powers[-1][:order],
        )

        return mat_skew_exp.view(lmat.shape)

    if use_autograd:
        return _exponential_orthogonalization_autograd(lmat)
    else:
        raise NotImplementedError


class LongRangeTransitionLayer(torch.nn.Module):
    config: LongRangeTransitionLayerConfig

    def __init__(self, config: LongRangeTransitionLayerConfig):
        torch.nn.Module.__init__(self)
        self.config = config

        if self.config.transition_dim > 1:
            self.inproj_bias = torch.nn.Parameter(
                torch.randn([config.num_heads, config.transition_dim, config.transition_dim])
            )
            self.inproj_weight = torch.nn.Parameter(
                torch.randn(
                    [
                        config.num_heads // config.sub_heads,
                        config.sub_heads,
                        config.transition_dim,
                        config.transition_dim,
                        config.input_dim // config.sub_heads,
                    ]
                )
            )
        self.eigenvalues_bias = torch.nn.Parameter(torch.randn([config.num_heads, config.transition_dim]))
        self.eigenvalues_weight = torch.nn.Parameter(
            torch.randn(
                [
                    config.num_heads // config.sub_heads,
                    config.sub_heads,
                    config.transition_dim,
                    config.input_dim // config.sub_heads,
                ]
            )
        )
        if not config.symmetric and self.config.transition_dim > 1:
            self.outproj_bias = torch.nn.Parameter(
                torch.randn([config.num_heads, config.transition_dim, config.transition_dim])
            )
            self.outproj_weight = torch.nn.Parameter(
                torch.randn(
                    [
                        config.num_heads // config.sub_heads,
                        config.sub_heads,
                        config.transition_dim,
                        config.transition_dim,
                        config.input_dim // config.sub_heads,
                    ]
                )
            )
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        with torch.no_grad():
            if self.config.transition_dim > 1:
                self.config.inproj_bias_init.instantiate(InitInterface)(self.inproj_bias)
                self.config.inproj_weight_init.instantiate(InitInterface)(self.inproj_weight)

                if not self.config.symmetric:
                    self.config.outproj_bias_init.instantiate(InitInterface)(self.outproj_bias)
                    self.config.outproj_weight_init.instantiate(InitInterface)(self.outproj_weight)

            self.config.eigenvalue_bias_init.instantiate(InitInterface)(self.eigenvalues_bias)
            self.config.eigenvalue_weight_init.instantiate(InitInterface)(self.eigenvalues_weight)

    def _eigenvalue_activation(self, x) -> torch.Tensor:
        if self.config.eigenvalue_representation == "logsigmoid":
            return torch.exp(self.config.eigenvalue_factor * F.logsigmoid(x))
        elif self.config.eigenvalue_representation == "expexp":
            return torch.exp(-self.config.eigenvalue_factor * torch.exp(-x))
        elif self.config.eigenvalue_representation == "tanh":
            return torch.tanh(self.config.eigenvalue_factor * x)

    def _normalize(self, mat):
        if self.config.normalization_mode == "qr":
            return torch.linalg.qr(mat.to(dtype=torch.float32))[0].to(dtype=mat.dtype)
        elif self.config.normalization_mode == "eigenvalue_restriction":
            return matrix_eigval_limit(
                mat,
                iterations=self.config.orthogonalization_order,
            )
        elif self.config.normalization_mode == "singularvalue_restriction":
            return matrix_singval_limit(mat, iterations=self.config.orthogonalization_order)
        elif self.config.normalization_mode == "householder_orthogonalization":
            return matrix_orthogonalize_householder(mat)
        elif self.config.normalization_mode == "exponential_orthogonalization":
            return matrix_orthogonalize_exponential(
                mat,
                factor=self.config.orthogonalization_factor,
                order=self.config.orthogonalization_order,
            )
        else:
            return ValueError("Bad normalization mode")

    def forward(self, x):
        x = x.reshape(*x.shape[:-1], self.config.sub_heads, self.config.input_dim // self.config.sub_heads)
        if self.config.weight:
            eigenvals = self.eigenvalues_bias + torch.einsum("nsbc,...sc->...nsb", self.eigenvalues_weight, x).reshape(
                *x.shape[:-2], self.config.num_heads, self.config.transition_dim
            )

        else:
            eigenvals = torch.tile(
                self.eigenvalues_bias.view(*((1,) * (x.ndim - 2)), *self.eigenvalues_bias.shape),
                x.shape[:-2] + (1, 1),
            )
        if self.config.transition_dim > 1:
            if self.config.weight:
                in_mat = self._normalize(
                    self.inproj_bias
                    + torch.einsum("nsbcd,...sd->...nsbc", self.inproj_weight, x).reshape(
                        *x.shape[:-2], self.config.num_heads, self.config.transition_dim, self.config.transition_dim
                    )
                )
            else:
                in_mat = self._normalize(
                    torch.tile(
                        self.inproj_bias.view(*((1,) * (x.ndim - 2)), *self.inproj_bias.shape),
                        x.shape[:-2] + (1, 1, 1),
                    )
                )
            if self.config.symmetric:
                out_mat = in_mat.transpose(-1, -2)
            else:
                if self.config.weight:
                    out_mat = self._normalize(
                        self.outproj_bias
                        + torch.einsum("nsbcd,...sd->...nsbc", self.outproj_weight, x).reshape(
                            *x.shape[:-2], self.config.num_heads, self.config.transition_dim, self.config.transition_dim
                        )
                    )
                else:
                    out_mat = self._normalize(
                        torch.tile(
                            self.outproj_bias.view(*((1,) * (x.ndim - 2)), *self.outproj_bias.shape),
                            x.shape[:-2] + (1, 1, 1),
                        )
                    )
            transition = torch.einsum(
                "...nab,...nb,...nbc->...nac",
                out_mat,
                self._eigenvalue_activation(eigenvals),
                in_mat,
            )
        else:
            transition = self._eigenvalue_activation(eigenvals)[..., None]
        return transition
