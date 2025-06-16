import jax
import jax.numpy as jnp
from flax import linen as nn
from ..util import log2
from ..config.longrange_transition_layer import (
    LongRangeTransitionLayerConfig,
)
from .dtype import str_dtype_to_jax
from .initialization import InitInterface


def matrix_eigval_limit(
    lmat,
    iterations=8,
    eigval_minlimit=1.0,
    eps_norm=1e-3,
    eps_mat=1e-3,
):
    """Limit eigenvalues of a matrix."""
    mat = lmat.reshape(-1, lmat.shape[-2], lmat.shape[-1])
    B, X, Y = mat.shape
    assert X == Y, "Matrix shape has to be square"
    mat_eye = mat - jnp.eye(X, dtype=mat.dtype)[None, :, :]
    x = 1.0 + jnp.zeros([B, X, 1], dtype=mat.dtype) + jnp.arange(X, dtype=mat.dtype)[None, :, None]

    def iteration_step(x, mat_eye):
        y = mat_eye @ x
        return y / (jnp.linalg.norm(y, axis=1, keepdims=True) + eps_norm)

    x = jax.lax.fori_loop(0, iterations, lambda i, x: iteration_step(x, mat_eye), x)

    eigval = jnp.einsum("...ji,...jk,...ki->...i", x, mat_eye, x)
    old_eigval = jnp.where(eigval < 0, 1.0 + eigval, 1.0)
    mat_eye2 = mat - old_eigval[:, None] * jnp.eye(X, dtype=mat.dtype)[None, :, :]

    x = jax.lax.fori_loop(0, iterations, lambda i, x: iteration_step(x, mat_eye2), x)

    eigval = jnp.maximum(
        jnp.einsum("...ji,...jk,...ki->...i", x, mat_eye2, x) + old_eigval,
        eigval_minlimit + jnp.zeros_like(old_eigval),
    )

    lmat_eye = mat_eye.reshape(*lmat.shape)
    mask = jnp.abs(lmat_eye).sum(axis=(-2, -1))[..., None, None] < eps_mat

    return jnp.where(
        mask,
        lmat,
        lmat / eigval.reshape(*lmat.shape[:-2])[..., None, None],
    )


def matrix_singval_limit(
    lmat,
    iterations=8,
    eigval_minlimit=1.0,
    eps_norm=1e-3,
    eps_mat=1e-3,
):
    """Limit singular values of a matrix."""
    mat = lmat.reshape(-1, lmat.shape[-2], lmat.shape[-1])
    B, X, Y = mat.shape
    m2 = jnp.matmul(mat, jnp.swapaxes(mat, -2, -1))
    assert X == Y, "Matrix shape has to be square"
    mat_eye = m2 - jnp.eye(X, dtype=mat.dtype)[None, :, :]
    x = 1.0 + jnp.zeros([B, X, 1], dtype=mat.dtype) + jnp.arange(X, dtype=mat.dtype)[None, :, None]

    def iteration_step(x, mat_eye):
        y = mat_eye @ x
        return y / (jnp.linalg.norm(y, axis=1, keepdims=True) + eps_norm)

    x = jax.lax.fori_loop(0, iterations, lambda i, x: iteration_step(x, mat_eye), x)

    eigval = jnp.einsum("...ji,...jk,...ki->...i", x, mat_eye, x)
    old_eigval = jnp.where(eigval < 0, 1.0 + eigval, 1.0)
    mat_eye2 = m2 - old_eigval[:, None] * jnp.eye(X, dtype=mat.dtype)[None, :, :]

    x = jax.lax.fori_loop(0, iterations, lambda i, x: iteration_step(x, mat_eye2), x)

    eigval = jnp.maximum(
        jnp.einsum("...ji,...jk,...ki->...i", x, mat_eye2, x) + old_eigval,
        eigval_minlimit + jnp.zeros_like(old_eigval),
    )

    lmat_eye = mat_eye.reshape(*lmat.shape)
    mask = jnp.abs(lmat_eye).sum(axis=(-2, -1))[..., None, None] < eps_mat

    singval = jnp.sqrt(eigval)

    return jnp.where(
        mask,
        lmat,
        lmat / singval.reshape(*lmat.shape[:-2])[..., None, None],
    )


def matrix_orthogonalize_householder(lmat: jax.Array, eps_norm: float = 1e-5):
    """Computes an orthogonal matrix from vectors via householder
    reflections."""
    mat = lmat.reshape(-1, lmat.shape[-2], lmat.shape[-1])
    B, X, Y = mat.shape
    mat = mat / (jnp.linalg.norm(mat, axis=-1, keepdims=True) + eps_norm)
    hhmat = jnp.eye(X, dtype=mat.dtype)[None, None, :, :] - 2 * jnp.einsum("nab,nac->nabc", mat, mat)

    def combine_reflections(hh):
        return jnp.einsum("nabc,nacd->nabd", hh[:, :-1:2], hh[:, 1::2])

    hhmats = [hhmat]
    for i in range(log2(X)):
        hhmats.append(combine_reflections(hhmats[-1]))

    return -hhmats[-1].reshape(lmat.shape)


def exponential_power_series(x: jax.Array, order: int = 8):
    orange = jnp.arange(order, dtype=x.dtype)
    lognorm = jnp.cumsum(jnp.where(orange > 0, jnp.log(orange), 0.0), axis=0)
    return jnp.exp(jnp.log(x.reshape(1, -1)) * orange[:, None] - lognorm[:, None]).reshape(order, *x.shape)


def matrix_orthogonalize_exponential(
    lmat: jax.Array,
    factor=0.1,
    order: int = 16,
    eps: float = 1e-3,
):
    """Calculates the approximation: exp(A) = 1 + A + 1/2*A**2 + 1/6*A**2"""
    mat = lmat.reshape(-1, lmat.shape[-2], lmat.shape[-1])
    B, X, Y = mat.shape
    assert X == Y, "Matrix has to be square"
    mat_skew = factor * (mat - jnp.swapaxes(mat, -2, -1))
    mat_max = jnp.sqrt(jnp.sum(mat_skew**2 + eps, axis=(-2, -1), keepdims=True))
    normalizer = jnp.maximum(mat_max, jnp.ones_like(mat_max))
    mat_skew_norm = mat_skew / normalizer
    identity_mat = jnp.eye(X, dtype=lmat.dtype)[None, :, :] + 0.0 * mat_skew

    if order == 1:
        return identity_mat.reshape(lmat.shape)

    def compute_power(mat, n):
        if n == 0:
            return identity_mat[None, :]
        prev = compute_power(mat, n - 1)
        curr = jnp.matmul(prev[-1:], mat[None, :])
        return jnp.concatenate([prev, curr], axis=0)

    powers = compute_power(mat_skew_norm, order - 1)
    mat_skew_exp = jnp.einsum(
        "x...,x...->...",
        exponential_power_series(normalizer, order),
        powers,
    )

    return mat_skew_exp.reshape(lmat.shape)


class LongRangeTransitionLayer(nn.Module):
    """Long range transition layer."""

    config: LongRangeTransitionLayerConfig

    def setup(self):
        param_dtype = str_dtype_to_jax(self.config.param_dtype)

        # Initialize parameters
        if self.config.transition_dim > 1:
            inproj_bias_shape = (self.config.num_heads, self.config.transition_dim, self.config.transition_dim)
            self.inproj_bias = self.param(
                "inproj_bias",
                self.config.inproj_bias_init.instantiate(InitInterface),
                inproj_bias_shape,
                param_dtype,
            )

            if self.config.weight:
                weight_shape = [
                    self.config.num_heads // self.config.sub_heads,
                    self.config.sub_heads,
                    self.config.transition_dim,
                    self.config.transition_dim,
                    self.config.input_dim // self.config.sub_heads,
                ]
                self.inproj_weight = self.param(
                    "inproj_weight",
                    self.config.inproj_weight_init.instantiate(InitInterface),
                    weight_shape,
                    param_dtype,
                )

        eigenvalues_bias_shape = [self.config.num_heads, self.config.transition_dim]
        self.eigenvalues_bias = self.param(
            "eigenvalues_bias",
            self.config.eigenvalue_bias_init.instantiate(InitInterface),
            eigenvalues_bias_shape,
            param_dtype,
        )

        if self.config.weight:
            weight_shape = [
                self.config.num_heads // self.config.sub_heads,
                self.config.sub_heads,
                self.config.transition_dim,
                self.config.input_dim // self.config.sub_heads,
            ]
            self.eigenvalues_weight = self.param(
                "eigenvalues_weight",
                self.config.eigenvalue_weight_init.instantiate(InitInterface),
                weight_shape,
                param_dtype,
            )

        # Initialize weights according to config
        if self.config.transition_dim > 1 and not self.config.symmetric:
            outproj_bias_shape = [self.config.num_heads, self.config.transition_dim, self.config.transition_dim]
            self.outproj_bias = self.param(
                "outproj_bias",
                self.config.outproj_bias_init.instantiate(InitInterface),
                outproj_bias_shape,
                param_dtype,
            )

            if self.config.weight:
                weight_shape = [
                    self.config.num_heads // self.config.sub_heads,
                    self.config.sub_heads,
                    self.config.transition_dim,
                    self.config.transition_dim,
                    self.config.input_dim // self.config.sub_heads,
                ]
                self.outproj_weight = self.param(
                    "outproj_weight",
                    self.config.outproj_weight_init.instantiate(InitInterface),
                    weight_shape,
                    param_dtype,
                )

    def _eigenvalue_activation(self, x) -> jax.Array:
        if self.config.eigenvalue_representation == "logsigmoid":
            return jnp.exp(self.config.eigenvalue_factor * jax.nn.log_sigmoid(x))
        elif self.config.eigenvalue_representation == "expexp":
            return jnp.exp(-self.config.eigenvalue_factor * jnp.exp(-x))
        elif self.config.eigenvalue_representation == "tanh":
            return jnp.tanh(self.config.eigenvalue_factor * x)

    def _normalize(self, mat):
        if self.config.normalization_mode == "qr":
            return jnp.linalg.qr(mat.astype(jnp.float32))[0].astype(mat.dtype)
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

    def __call__(self, x):
        # Cast input to computation dtype
        dtype = str_dtype_to_jax(self.config.dtype)
        x = x.astype(dtype)
        x = x.reshape(*x.shape[:-1], self.config.sub_heads, self.config.input_dim // self.config.sub_heads)

        if self.config.weight:
            eigenvals = (
                self.eigenvalues_bias
                + jnp.einsum("nsbc,...sc->...nsb", self.eigenvalues_weight, x).reshape(
                    *x.shape[:-2], self.config.num_heads, self.config.transition_dim
                )
            ).astype(dtype)
        else:
            eigenvals = jnp.tile(
                self.eigenvalues_bias.astype(dtype).reshape(*((1,) * (x.ndim - 2)), *self.eigenvalues_bias.shape),
                x.shape[:-2] + (1, 1),
            )
        if self.config.transition_dim > 1:
            if self.config.weight:
                in_mat = self._normalize(
                    self.inproj_bias.astype(dtype)
                    + jnp.einsum("nsbcd,...sd->...nsbc", self.inproj_weight.astype(dtype), x).reshape(
                        *x.shape[:-2], self.config.num_heads, self.config.transition_dim, self.config.transition_dim
                    )
                )
                if self.config.symmetric:
                    out_mat = jnp.swapaxes(in_mat, -1, -2)
                else:
                    out_mat = self._normalize(
                        self.outproj_bias.astype(dtype)
                        + jnp.einsum("nsbcd,...sd->...nsbc", self.outproj_weight.astype(dtype), x).reshape(
                            *x.shape[:-2], self.config.num_heads, self.config.transition_dim, self.config.transition_dim
                        )
                    )
            else:
                in_mat = self._normalize(
                    jnp.tile(
                        self.inproj_bias.astype(dtype).reshape(*((1,) * (x.ndim - 2)), *self.inproj_bias.shape),
                        x.shape[:-2] + (1, 1, 1),
                    )
                )
                if self.config.symmetric:
                    out_mat = jnp.swapaxes(in_mat, -1, -2)
                else:
                    out_mat = self._normalize(
                        jnp.tile(
                            self.outproj_bias.astype(dtype).reshape(*((1,) * (x.ndim - 2)), *self.outproj_bias.shape),
                            x.shape[:-2] + (1, 1, 1),
                        )
                    )
            transition = jnp.einsum(
                "...nab,...nb,...nbc->...nac",
                out_mat,
                self._eigenvalue_activation(eigenvals),
                in_mat,
            )
        else:
            transition = self._eigenvalue_activation(eigenvals)[..., None]
        return transition
