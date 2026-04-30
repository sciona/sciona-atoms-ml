"""MinCovDet covariance-correction helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mincovdet_correct_covariance_guard,
    witness_mincovdet_corrected_covariance,
    witness_mincovdet_corrected_distances,
    witness_mincovdet_empirical_correction_factor,
)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _bool_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.bool_)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1)


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


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _positive_finite_scalar(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0


def _guard_inputs_valid(raw_covariance: object, raw_support: object, raw_distances: object) -> bool:
    support = np.asarray(raw_support, dtype=np.bool_)
    distances = np.asarray(raw_distances, dtype=np.float64)
    return bool(
        _finite_square_matrix(raw_covariance)
        and _bool_vector(raw_support)
        and _finite_vector(raw_distances)
        and support.shape == distances.shape
    )


def _guard_result_valid(result: object) -> bool:
    return isinstance(result, bool)


def _correction_factor_valid(result: object) -> bool:
    return _positive_finite_scalar(result)


def _corrected_covariance_valid(result: object, raw_covariance: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    covariance = np.asarray(raw_covariance, dtype=np.float64)
    return bool(
        values.shape == covariance.shape
        and np.all(np.isfinite(values))
        and np.allclose(values, values.T)
    )


def _corrected_distances_valid(result: object, raw_distances: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    distances = np.asarray(raw_distances, dtype=np.float64)
    return bool(values.shape == distances.shape and np.all(np.isfinite(values)) and np.all(values >= 0.0))


@register_atom(witness_mincovdet_correct_covariance_guard)
@icontract.require(
    lambda raw_covariance, raw_support, raw_distances: _guard_inputs_valid(raw_covariance, raw_support, raw_distances),
    "raw_covariance, raw_support, and raw_distances must be compatible finite MinCovDet raw state",
)
@icontract.ensure(lambda result: _guard_result_valid(result), "guard result must be boolean")
def mincovdet_correct_covariance_guard(
    raw_covariance: NDArray[np.float64],
    raw_support: NDArray[np.bool_],
    raw_distances: NDArray[np.float64],
) -> bool:
    """Apply MinCovDet's zero-raw-covariance guard before empirical correction."""
    covariance = np.asarray(raw_covariance, dtype=np.float64)
    support = np.asarray(raw_support, dtype=np.bool_)
    distances = np.asarray(raw_distances, dtype=np.float64)
    n_samples = int(distances.shape[0])
    n_support = int(np.sum(support))
    if n_support < n_samples and np.allclose(covariance, 0.0):
        raise ValueError(
            "The covariance matrix of the support data is equal to 0, try to increase support_fraction"
        )
    return True


@register_atom(witness_mincovdet_empirical_correction_factor)
@icontract.require(lambda raw_distances: _finite_vector(raw_distances) and np.all(np.asarray(raw_distances, dtype=np.float64) >= 0.0), "raw_distances must be a finite nonnegative vector")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.ensure(lambda result: _correction_factor_valid(result), "correction factor must be positive and finite")
def mincovdet_empirical_correction_factor(
    raw_distances: NDArray[np.float64],
    n_features: int,
) -> float:
    """Compute MinCovDet's empirical covariance correction factor from raw distances."""
    distances = np.asarray(raw_distances, dtype=np.float64)
    return float(np.median(distances) / chi2(int(n_features)).isf(0.5))


@register_atom(witness_mincovdet_corrected_covariance)
@icontract.require(lambda raw_covariance: _finite_square_matrix(raw_covariance), "raw_covariance must be a finite square matrix")
@icontract.require(lambda correction_factor: _positive_finite_scalar(correction_factor), "correction_factor must be positive and finite")
@icontract.ensure(lambda result, raw_covariance: _corrected_covariance_valid(result, raw_covariance), "corrected covariance must preserve the raw covariance shape")
def mincovdet_corrected_covariance(
    raw_covariance: NDArray[np.float64],
    correction_factor: float,
) -> NDArray[np.float64]:
    """Scale raw covariance by MinCovDet's empirical correction factor."""
    return np.asarray(np.asarray(raw_covariance, dtype=np.float64) * float(correction_factor), dtype=np.float64)


@register_atom(witness_mincovdet_corrected_distances)
@icontract.require(lambda raw_distances: _finite_vector(raw_distances) and np.all(np.asarray(raw_distances, dtype=np.float64) >= 0.0), "raw_distances must be a finite nonnegative vector")
@icontract.require(lambda correction_factor: _positive_finite_scalar(correction_factor), "correction_factor must be positive and finite")
@icontract.ensure(lambda result, raw_distances: _corrected_distances_valid(result, raw_distances), "corrected distances must preserve the raw distance shape")
def mincovdet_corrected_distances(
    raw_distances: NDArray[np.float64],
    correction_factor: float,
) -> NDArray[np.float64]:
    """Scale raw squared distances by the inverse MinCovDet empirical correction factor."""
    return np.asarray(np.asarray(raw_distances, dtype=np.float64) / float(correction_factor), dtype=np.float64)
