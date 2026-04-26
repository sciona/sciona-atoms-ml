"""Dense pairwise-kernel helper atoms adapted from sklearn.metrics.pairwise."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_pairwise_cosine_similarity,
    witness_pairwise_default_gamma,
    witness_pairwise_laplacian_kernel,
    witness_pairwise_linear_kernel,
    witness_pairwise_polynomial_kernel,
    witness_pairwise_sigmoid_kernel,
)


def _matrix_2d(X: object) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2)


def _optional_matrix_2d(Y: object) -> bool:
    return bool(Y is None or _matrix_2d(Y))


def _finite_matrix(X: object) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and np.all(np.isfinite(values)))


def _same_feature_count(X: object, Y: object | None) -> bool:
    if Y is None:
        return True
    return bool(np.asarray(X).ndim == 2 and np.asarray(Y).ndim == 2 and np.asarray(X).shape[1] == np.asarray(Y).shape[1])


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_scalar_or_none(value: object) -> bool:
    return bool(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0))


def _positive_scalar_or_none(value: object) -> bool:
    return bool(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0))


def _finite_scalar(value: object) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _kernel_matrix_valid(result: NDArray[np.float64], X: object, Y: object | None) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_rows = np.asarray(X).shape[0]
    n_cols = n_rows if Y is None else np.asarray(Y).shape[0]
    return bool(values.shape == (n_rows, n_cols) and np.all(np.isfinite(values)))


def _cosine_result_valid(result: NDArray[np.float64], X: object, Y: object | None) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(_kernel_matrix_valid(values, X, Y) and np.all(values >= -1.0 - 1e-12) and np.all(values <= 1.0 + 1e-12))


def _default_gamma(n_features: int, gamma: float | None) -> float:
    return 1.0 / float(n_features) if gamma is None else float(gamma)


@register_atom(witness_pairwise_default_gamma)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(lambda gamma: _nonnegative_scalar_or_none(gamma), "gamma must be None or a non-negative finite scalar")
@icontract.ensure(lambda result: isinstance(result, float) and np.isfinite(result) and result >= 0.0, "resolved gamma must be finite and non-negative")
def pairwise_default_gamma(
    n_features: int,
    gamma: float | None = None,
) -> float:
    """Resolve sklearn's default gamma fallback of 1 / n_features."""
    return float(_default_gamma(int(n_features), gamma))


@register_atom(witness_pairwise_linear_kernel)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y) and (Y is None or _finite_matrix(Y)), "Y must be None or a finite 2D matrix")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.ensure(lambda result, X, Y: _kernel_matrix_valid(result, X, Y), "kernel matrix must match sample counts")
def pairwise_linear_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute the dense linear kernel X @ Y.T."""
    checked_x = np.asarray(X, dtype=np.float64)
    checked_y = checked_x if Y is None else np.asarray(Y, dtype=np.float64)
    return np.asarray(checked_x @ checked_y.T, dtype=np.float64)


@register_atom(witness_pairwise_polynomial_kernel)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y) and (Y is None or _finite_matrix(Y)), "Y must be None or a finite 2D matrix")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda degree: _finite_scalar(degree) and float(degree) >= 1.0, "degree must be finite and at least 1")
@icontract.require(lambda gamma: _nonnegative_scalar_or_none(gamma), "gamma must be None or a non-negative finite scalar")
@icontract.require(lambda coef0: _finite_scalar(coef0), "coef0 must be finite")
@icontract.ensure(lambda result, X, Y: _kernel_matrix_valid(result, X, Y), "kernel matrix must match sample counts")
def pairwise_polynomial_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    degree: float = 3.0,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the dense polynomial kernel (gamma * <X, Y> + coef0) ** degree."""
    checked_x = np.asarray(X, dtype=np.float64)
    checked_y = checked_x if Y is None else np.asarray(Y, dtype=np.float64)
    resolved_gamma = _default_gamma(checked_x.shape[1], gamma)
    kernel = pairwise_linear_kernel(checked_x, checked_y)
    kernel *= resolved_gamma
    kernel += float(coef0)
    kernel **= float(degree)
    return np.asarray(kernel, dtype=np.float64)


@register_atom(witness_pairwise_laplacian_kernel)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y) and (Y is None or _finite_matrix(Y)), "Y must be None or a finite 2D matrix")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda gamma: _positive_scalar_or_none(gamma), "gamma must be None or a strictly positive finite scalar")
@icontract.ensure(lambda result, X, Y: _kernel_matrix_valid(result, X, Y), "kernel matrix must match sample counts")
def pairwise_laplacian_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    gamma: float | None = None,
) -> NDArray[np.float64]:
    """Compute the dense Laplacian kernel exp(-gamma * ||x - y||_1)."""
    checked_x = np.asarray(X, dtype=np.float64)
    checked_y = checked_x if Y is None else np.asarray(Y, dtype=np.float64)
    resolved_gamma = _default_gamma(checked_x.shape[1], gamma)
    distances = cdist(checked_x, checked_y, metric="cityblock")
    kernel = -resolved_gamma * distances
    np.exp(kernel, out=kernel)
    return np.asarray(kernel, dtype=np.float64)


@register_atom(witness_pairwise_sigmoid_kernel)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y) and (Y is None or _finite_matrix(Y)), "Y must be None or a finite 2D matrix")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda gamma: _nonnegative_scalar_or_none(gamma), "gamma must be None or a non-negative finite scalar")
@icontract.require(lambda coef0: _finite_scalar(coef0), "coef0 must be finite")
@icontract.ensure(lambda result, X, Y: _kernel_matrix_valid(result, X, Y), "kernel matrix must match sample counts")
def pairwise_sigmoid_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the dense sigmoid kernel tanh(gamma * <X, Y> + coef0)."""
    checked_x = np.asarray(X, dtype=np.float64)
    checked_y = checked_x if Y is None else np.asarray(Y, dtype=np.float64)
    resolved_gamma = _default_gamma(checked_x.shape[1], gamma)
    kernel = pairwise_linear_kernel(checked_x, checked_y)
    kernel *= resolved_gamma
    kernel += float(coef0)
    np.tanh(kernel, out=kernel)
    return np.asarray(kernel, dtype=np.float64)


@register_atom(witness_pairwise_cosine_similarity)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y) and (Y is None or _finite_matrix(Y)), "Y must be None or a finite 2D matrix")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.ensure(lambda result, X, Y: _cosine_result_valid(result, X, Y), "cosine-similarity matrix must match sample counts and remain within [-1, 1]")
def pairwise_cosine_similarity(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute dense cosine similarity with sklearn's zero-row behavior."""
    checked_x = np.asarray(X, dtype=np.float64)
    checked_y = checked_x if Y is None else np.asarray(Y, dtype=np.float64)
    x_norms = np.linalg.norm(checked_x, axis=1)
    y_norms = x_norms if Y is None else np.linalg.norm(checked_y, axis=1)
    denom = x_norms[:, None] * y_norms[None, :]
    numer = checked_x @ checked_y.T
    similarities = np.divide(
        numer,
        denom,
        out=np.zeros_like(numer, dtype=np.float64),
        where=denom != 0.0,
    )
    return np.asarray(similarities, dtype=np.float64)
