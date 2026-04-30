"""Dense distance-kernel helper atoms adapted from sklearn.metrics.pairwise."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_pairwise_additive_chi2_kernel,
    witness_pairwise_chi2_kernel,
    witness_pairwise_rbf_kernel,
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


def _strictly_positive_scalar_or_none(value: object) -> bool:
    return bool(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0))


def _strictly_positive_scalar(value: object) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)


def _nonnegative_matrix(X: object) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _kernel_matrix_valid(result: NDArray[np.float64], X: object, Y: object | None) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_rows = np.asarray(X).shape[0]
    n_cols = n_rows if Y is None else np.asarray(Y).shape[0]
    return bool(values.shape == (n_rows, n_cols) and np.all(np.isfinite(values)))


def _rbf_kernel_valid(result: NDArray[np.float64], X: object, Y: object | None) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(_kernel_matrix_valid(values, X, Y) and np.all(values > 0.0) and np.all(values <= 1.0))


def _additive_chi2_valid(result: NDArray[np.float64], X: object, Y: object | None) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(_kernel_matrix_valid(values, X, Y) and np.all(values <= 0.0))


def _chi2_valid(result: NDArray[np.float64], X: object, Y: object | None) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(_kernel_matrix_valid(values, X, Y) and np.all(values > 0.0) and np.all(values <= 1.0))


@register_atom(witness_pairwise_rbf_kernel)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y) and (Y is None or _finite_matrix(Y)), "Y must be None or a finite 2D matrix")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda gamma: _strictly_positive_scalar_or_none(gamma), "gamma must be None or a strictly positive finite scalar")
@icontract.ensure(lambda result, X, Y: _rbf_kernel_valid(result, X, Y), "RBF kernel matrix must match sample counts and remain within (0, 1]")
def pairwise_rbf_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    gamma: float | None = None,
) -> NDArray[np.float64]:
    """Compute the dense radial-basis-function kernel exp(-gamma * ||x-y||^2)."""
    checked_x = np.asarray(X, dtype=np.float64)
    checked_y = checked_x if Y is None else np.asarray(Y, dtype=np.float64)
    resolved_gamma = 1.0 / checked_x.shape[1] if gamma is None else float(gamma)
    squared_distances = cdist(checked_x, checked_y, metric="sqeuclidean")
    kernel = -resolved_gamma * squared_distances
    np.exp(kernel, out=kernel)
    return np.asarray(kernel, dtype=np.float64)


@register_atom(witness_pairwise_additive_chi2_kernel)
@icontract.require(lambda X: _nonnegative_matrix(X), "X must be a finite nonnegative 2D matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y) and (Y is None or _nonnegative_matrix(Y)), "Y must be None or a finite nonnegative 2D matrix")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.ensure(lambda result, X, Y: _additive_chi2_valid(result, X, Y), "additive chi-squared kernel matrix must match sample counts and remain nonpositive")
def pairwise_additive_chi2_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute the dense additive chi-squared kernel as the negative normalized squared difference sum."""
    checked_x = np.asarray(X, dtype=np.float64)
    checked_y = checked_x if Y is None else np.asarray(Y, dtype=np.float64)
    xb = checked_x[:, None, :]
    yb = checked_y[None, :, :]
    numer = -((xb - yb) ** 2)
    denom = xb + yb
    terms = np.divide(
        numer,
        denom,
        out=np.zeros_like(numer, dtype=np.float64),
        where=denom != 0.0,
    )
    return np.asarray(np.sum(terms, axis=2), dtype=np.float64)


@register_atom(witness_pairwise_chi2_kernel)
@icontract.require(lambda X: _nonnegative_matrix(X), "X must be a finite nonnegative 2D matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y) and (Y is None or _nonnegative_matrix(Y)), "Y must be None or a finite nonnegative 2D matrix")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda gamma: _strictly_positive_scalar(gamma), "gamma must be a strictly positive finite scalar")
@icontract.ensure(lambda result, X, Y: _chi2_valid(result, X, Y), "chi-squared kernel matrix must match sample counts and remain within (0, 1]")
def pairwise_chi2_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    gamma: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the dense exponentiated chi-squared kernel from nonnegative inputs."""
    additive = pairwise_additive_chi2_kernel(X, Y)
    scaled = np.asarray(additive, dtype=np.float64) * float(gamma)
    np.exp(scaled, out=scaled)
    return np.asarray(scaled, dtype=np.float64)
