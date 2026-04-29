"""MinCovDet fit-bookkeeping helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from sciona.atoms.ml.sklearn.covariance import empirical_covariance
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mincovdet_assume_centered_raw_covariance,
    witness_mincovdet_assume_centered_raw_distances,
    witness_mincovdet_assume_centered_raw_location,
    witness_mincovdet_full_rank_warning_required,
)


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 2
        and array.shape[1] >= 1
        and np.all(np.isfinite(array))
    )


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _support_mask_valid(mask: object, n_samples: int) -> bool:
    try:
        values = np.asarray(mask, dtype=np.bool_)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape == (n_samples,) and np.any(values))


def _square_matrix_valid(values: object) -> bool:
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


def _warning_result_valid(result: object) -> bool:
    return isinstance(result, bool)


def _location_result_valid(result: object, n_features: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (n_features,) and np.all(values == 0.0))


def _covariance_result_valid(result: object, X: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_features = np.asarray(X, dtype=np.float64).shape[1]
    return bool(values.shape == (n_features, n_features) and np.all(np.isfinite(values)) and np.allclose(values, values.T))


def _distance_result_valid(result: object, X: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_samples = np.asarray(X, dtype=np.float64).shape[0]
    return bool(values.shape == (n_samples,) and np.all(np.isfinite(values)) and np.all(values >= 0.0))


@register_atom(witness_mincovdet_full_rank_warning_required)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D data matrix with at least two samples")
@icontract.require(lambda tol: isinstance(tol, (int, float)) and not isinstance(tol, bool) and np.isfinite(float(tol)) and float(tol) > 0.0, "tol must be a positive finite scalar")
@icontract.ensure(lambda result: _warning_result_valid(result), "warning predicate must be boolean")
def mincovdet_full_rank_warning_required(
    X: NDArray[np.float64],
    *,
    tol: float = 1e-8,
) -> bool:
    """Return whether MinCovDet.fit would warn that `X.T @ X` is not full rank."""
    values = np.asarray(X, dtype=np.float64)
    n_features = values.shape[1]
    rank = int((linalg.svdvals(np.dot(values.T, values)) > float(tol)).sum())
    return rank != n_features


@register_atom(witness_mincovdet_assume_centered_raw_location)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.ensure(lambda result, n_features: _location_result_valid(result, n_features), "raw location must be an all-zero feature vector")
def mincovdet_assume_centered_raw_location(
    n_features: int,
) -> NDArray[np.float64]:
    """Return MinCovDet's zero raw-location vector for `assume_centered=True`."""
    return np.zeros(int(n_features), dtype=np.float64)


@register_atom(witness_mincovdet_assume_centered_raw_covariance)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D data matrix with at least two samples")
@icontract.require(lambda X, raw_support: _support_mask_valid(raw_support, np.asarray(X, dtype=np.float64).shape[0]), "raw_support must be a boolean mask over samples with at least one selected row")
@icontract.ensure(lambda result, X: _covariance_result_valid(result, X), "raw covariance must be a finite symmetric square matrix")
def mincovdet_assume_centered_raw_covariance(
    X: NDArray[np.float64],
    raw_support: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Recompute MinCovDet's raw covariance from the raw support when centered at zero."""
    values = np.asarray(X, dtype=np.float64)
    support = np.asarray(raw_support, dtype=np.bool_)
    return np.asarray(empirical_covariance(values[support], assume_centered=True), dtype=np.float64)


@register_atom(witness_mincovdet_assume_centered_raw_distances)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D data matrix with at least two samples")
@icontract.require(lambda raw_covariance: _square_matrix_valid(raw_covariance), "raw_covariance must be a finite square matrix")
@icontract.require(lambda X, raw_covariance: np.asarray(raw_covariance, dtype=np.float64).shape == (np.asarray(X, dtype=np.float64).shape[1], np.asarray(X, dtype=np.float64).shape[1]), "raw_covariance must match X feature dimensions")
@icontract.ensure(lambda result, X: _distance_result_valid(result, X), "raw distances must be a nonnegative finite vector over samples")
def mincovdet_assume_centered_raw_distances(
    X: NDArray[np.float64],
    raw_covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute MinCovDet's raw squared distances when `assume_centered=True`."""
    values = np.asarray(X, dtype=np.float64)
    covariance = np.asarray(raw_covariance, dtype=np.float64)
    precision = linalg.pinvh(covariance)
    return np.asarray(np.sum(np.dot(values, precision) * values, axis=1), dtype=np.float64)
