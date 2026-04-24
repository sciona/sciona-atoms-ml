"""Gaussian-process regression log-marginal-likelihood gradient atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_solve

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_log_marginal_gradient,
    witness_gp_log_marginal_gradient_dims,
    witness_gp_log_marginal_gradient_inner_term,
)


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim in {1, 2}
        and array.shape[0] >= 1
        and (array.ndim == 1 or array.shape[1] >= 1)
        and np.all(np.isfinite(array))
    )


def _finite_square_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
    )


def _lower_cholesky_factor(values: object) -> bool:
    if not _finite_square_matrix(values):
        return False
    factor = np.asarray(values, dtype=np.float64)
    return bool(np.allclose(factor, np.tril(factor)) and np.all(np.diag(factor) > 0.0))


def _targets_compatible(L: NDArray[np.float64], dual_coefficients: NDArray[np.float64]) -> bool:
    if not (_lower_cholesky_factor(L) and _finite_matrix(dual_coefficients)):
        return False
    alpha = np.asarray(dual_coefficients, dtype=np.float64)
    return bool(alpha.shape[0] == np.asarray(L, dtype=np.float64).shape[0])


def _inner_term_valid(result: NDArray[np.float64], L: NDArray[np.float64], dual_coefficients: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    alpha = np.asarray(dual_coefficients, dtype=np.float64)
    n_samples = np.asarray(L, dtype=np.float64).shape[0]
    n_targets = 1 if alpha.ndim == 1 else alpha.shape[1]
    return bool(values.shape == (n_samples, n_samples, n_targets) and np.all(np.isfinite(values)))


def _kernel_gradient_valid(kernel_gradient: object, n_samples: int) -> bool:
    try:
        gradient = np.asarray(kernel_gradient, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        gradient.ndim == 3
        and gradient.shape[0] == n_samples
        and gradient.shape[1] == n_samples
        and gradient.shape[2] >= 1
        and np.all(np.isfinite(gradient))
    )


def _gradient_dims_inputs_valid(inner_term: NDArray[np.float64], kernel_gradient: NDArray[np.float64]) -> bool:
    tensor = np.asarray(inner_term, dtype=np.float64)
    return bool(
        tensor.ndim == 3
        and tensor.shape[0] >= 1
        and tensor.shape[0] == tensor.shape[1]
        and _kernel_gradient_valid(kernel_gradient, tensor.shape[0])
    )


def _gradient_dims_valid(result: NDArray[np.float64], inner_term: NDArray[np.float64], kernel_gradient: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    tensor = np.asarray(inner_term, dtype=np.float64)
    gradient = np.asarray(kernel_gradient, dtype=np.float64)
    return bool(values.shape == (gradient.shape[2], tensor.shape[2]) and np.all(np.isfinite(values)))


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _gradient_valid(result: NDArray[np.float64], gradient_dims: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    dims = np.asarray(gradient_dims, dtype=np.float64)
    return bool(values.shape == (dims.shape[0],) and np.all(np.isfinite(values)))


@register_atom(witness_gp_log_marginal_gradient_inner_term)
@icontract.require(lambda L, dual_coefficients: _targets_compatible(L, dual_coefficients), "L must be a lower Cholesky factor and dual_coefficients must align with its sample count")
@icontract.ensure(lambda result, L, dual_coefficients: _inner_term_valid(result, L, dual_coefficients), "inner term must be a finite sample-by-sample-by-target tensor")
def gp_log_marginal_gradient_inner_term(
    L: NDArray[np.float64],
    dual_coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build sklearn's shared log-marginal-likelihood gradient inner term."""
    alpha = np.asarray(dual_coefficients, dtype=np.float64)
    if alpha.ndim == 1:
        alpha = alpha[:, np.newaxis]
    inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
    kernel_inverse = cho_solve((np.asarray(L, dtype=np.float64), True), np.eye(alpha.shape[0]), check_finite=False)
    inner_term = inner_term - kernel_inverse[..., np.newaxis]
    return np.asarray(inner_term, dtype=np.float64)


@register_atom(witness_gp_log_marginal_gradient_dims)
@icontract.require(lambda inner_term, kernel_gradient: _gradient_dims_inputs_valid(inner_term, kernel_gradient), "inner_term and kernel_gradient must be finite tensors with matching sample axes")
@icontract.ensure(lambda result, inner_term, kernel_gradient: _gradient_dims_valid(result, inner_term, kernel_gradient), "gradient dimensions must be finite with one row per kernel parameter and one column per target")
def gp_log_marginal_gradient_dims(
    inner_term: NDArray[np.float64],
    kernel_gradient: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute per-parameter per-output GP log-marginal-likelihood gradient terms."""
    return np.asarray(
        0.5 * np.einsum(
            "ijl,jik->kl",
            np.asarray(inner_term, dtype=np.float64),
            np.asarray(kernel_gradient, dtype=np.float64),
        ),
        dtype=np.float64,
    )


@register_atom(witness_gp_log_marginal_gradient)
@icontract.require(lambda gradient_dims: _finite_matrix(gradient_dims), "gradient_dims must be a finite 1D or 2D array")
@icontract.ensure(lambda result, gradient_dims: _gradient_valid(result, gradient_dims), "gradient must be a finite vector with one entry per kernel parameter")
def gp_log_marginal_gradient(
    gradient_dims: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum per-output GP log-marginal-likelihood gradient terms across targets."""
    dims = np.asarray(gradient_dims, dtype=np.float64)
    if dims.ndim == 1:
        return np.asarray(dims, dtype=np.float64)
    return np.asarray(dims.sum(axis=-1), dtype=np.float64)
