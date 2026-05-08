"""Deterministic FastMCD c-step helper atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from sciona.atoms.ml.sklearn.covariance import empirical_covariance
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_fast_mcd_c_step,
    witness_fast_mcd_initial_random_support_indices,
    witness_fast_mcd_support_indices_from_estimates,
    witness_fast_mcd_support_statistics,
)

RandomStateLike = int | np.random.RandomState | None
SupportStats = tuple[NDArray[np.float64], NDArray[np.float64], float]
CStepResult = tuple[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.bool_], NDArray[np.float64]]

def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 1 and np.all(np.isfinite(array)))

def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))

def _finite_square_matrix(values: object) -> bool:
    return bool(_finite_matrix(values) and np.asarray(values, dtype=np.float64).shape[0] == np.asarray(values, dtype=np.float64).shape[1])

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _support_size_valid(n_support: int, n_samples: int) -> bool:
    return bool(_positive_int(n_support) and n_support <= n_samples)

def _support_indices_valid(support_indices: object, n_samples: int, n_support: int | None = None) -> bool:
    array = np.asarray(support_indices)
    return bool(
        array.ndim == 1
        and (n_support is None or array.shape[0] == n_support)
        and array.shape[0] >= 1
        and np.issubdtype(array.dtype, np.integer)
        and np.all(array >= 0)
        and np.all(array < n_samples)
        and np.unique(array).shape[0] == array.shape[0]
    )

def _estimates_valid(X: NDArray[np.float64], location: NDArray[np.float64], covariance: NDArray[np.float64], n_support: int) -> bool:
    X_values = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X_values.shape
    return bool(
        _finite_matrix(X)
        and _finite_vector(location)
        and _finite_square_matrix(covariance)
        and location.shape == (n_features,)
        and covariance.shape == (n_features, n_features)
        and _support_size_valid(n_support, n_samples)
    )

def _det_valid(det: float) -> bool:
    return bool(isinstance(det, (int, float, np.floating)) and (np.isfinite(float(det)) or np.isinf(float(det))))

def _support_stats_valid(result: SupportStats, X: NDArray[np.float64], support_indices: NDArray[np.int64]) -> bool:
    if not (isinstance(result, tuple) and len(result) == 3):
        return False
    location, covariance, det = result
    X_values = np.asarray(X, dtype=np.float64)
    support_values = np.asarray(support_indices, dtype=np.int64)
    return bool(
        np.asarray(location, dtype=np.float64).shape == (X_values.shape[1],)
        and np.asarray(covariance, dtype=np.float64).shape == (X_values.shape[1], X_values.shape[1])
        and np.all(np.isfinite(np.asarray(location, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(covariance, dtype=np.float64)))
        and np.allclose(np.asarray(covariance, dtype=np.float64), np.asarray(covariance, dtype=np.float64).T)
        and _det_valid(det)
        and support_values.shape[0] >= 1
    )

def _c_step_result_valid(result: CStepResult, X: NDArray[np.float64]) -> bool:
    if not (isinstance(result, tuple) and len(result) == 5):
        return False
    location, covariance, det, support, dist = result
    X_values = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X_values.shape
    return bool(
        np.asarray(location, dtype=np.float64).shape == (n_features,)
        and np.asarray(covariance, dtype=np.float64).shape == (n_features, n_features)
        and _det_valid(det)
        and np.asarray(support).shape == (n_samples,)
        and np.asarray(support).dtype == np.bool_
        and np.any(np.asarray(support))
        and np.asarray(dist, dtype=np.float64).shape == (n_samples,)
        and np.all(np.isfinite(np.asarray(dist, dtype=np.float64)))
        and np.all(np.asarray(dist, dtype=np.float64) >= 0.0)
    )

@register_atom(witness_fast_mcd_initial_random_support_indices)
@icontract.require(lambda n_samples: _positive_int(n_samples) and n_samples >= 2, "n_samples must be at least 2")
@icontract.require(lambda n_support, n_samples: _support_size_valid(n_support, n_samples), "n_support must lie in [1, n_samples]")
@icontract.ensure(lambda result, n_samples, n_support: _support_indices_valid(result, n_samples, n_support), "support indices must be unique integers inside the sample range")
def fast_mcd_initial_random_support_indices(
    n_samples: int,
    n_support: int,
    *,
    random_state: RandomStateLike = None,
) -> NDArray[np.int64]:
    from sklearn.utils import check_random_state
    """Draw sklearn's initial random FastMCD support indices."""
    rng = check_random_state(random_state)
    return np.asarray(rng.permutation(int(n_samples))[: int(n_support)], dtype=np.int64)

@register_atom(witness_fast_mcd_support_indices_from_estimates)
@icontract.require(lambda X, location, covariance, n_support: _estimates_valid(X, location, covariance, n_support), "X, location, covariance, and n_support must be compatible")
@icontract.ensure(lambda result, X, n_support: _support_indices_valid(result, np.asarray(X, dtype=np.float64).shape[0], n_support), "support indices must be unique integers inside the sample range")
def fast_mcd_support_indices_from_estimates(
    X: NDArray[np.float64],
    location: NDArray[np.float64],
    covariance: NDArray[np.float64],
    *,
    n_support: int,
) -> NDArray[np.int64]:
    """Choose FastMCD support indices from supplied location and covariance estimates."""
    X_values = np.asarray(X, dtype=np.float64)
    location_values = np.asarray(location, dtype=np.float64)
    covariance_values = np.asarray(covariance, dtype=np.float64)
    precision = linalg.pinvh(covariance_values)
    centered = X_values - location_values
    dist = np.asarray(np.sum(np.dot(centered, precision) * centered, axis=1), dtype=np.float64)
    return np.asarray(np.argpartition(dist, int(n_support) - 1)[: int(n_support)], dtype=np.int64)

@register_atom(witness_fast_mcd_support_statistics)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D sample matrix")
@icontract.require(lambda X, support_indices: _support_indices_valid(support_indices, np.asarray(X, dtype=np.float64).shape[0]), "support_indices must be unique integers inside the sample range")
@icontract.ensure(lambda result, X, support_indices: _support_stats_valid(result, X, support_indices), "support statistics must return a valid location, covariance, and determinant")
def fast_mcd_support_statistics(
    X: NDArray[np.float64],
    support_indices: NDArray[np.int64],
) -> SupportStats:
    from sklearn.utils.extmath import fast_logdet
    """Compute FastMCD location, covariance, and log-determinant from a support set."""
    X_values = np.asarray(X, dtype=np.float64)
    indices = np.asarray(support_indices, dtype=np.int64)
    X_support = X_values[indices]
    location = np.asarray(X_support.mean(axis=0), dtype=np.float64)
    covariance = np.asarray(empirical_covariance(X_support), dtype=np.float64)
    det = float(fast_logdet(covariance))
    return location, covariance, det

@register_atom(witness_fast_mcd_c_step)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D sample matrix")
@icontract.require(lambda X, n_support: _support_size_valid(n_support, np.asarray(X, dtype=np.float64).shape[0]), "n_support must lie in [1, n_samples]")
@icontract.require(lambda remaining_iterations: isinstance(remaining_iterations, int) and not isinstance(remaining_iterations, bool) and remaining_iterations >= 0, "remaining_iterations must be a nonnegative integer")
@icontract.require(lambda X, initial_location, initial_covariance, n_support: (initial_location is None and initial_covariance is None) or (initial_location is not None and initial_covariance is not None and _estimates_valid(X, initial_location, initial_covariance, n_support)), "initial_location and initial_covariance must be supplied together and match X")
@icontract.ensure(lambda result, X: _c_step_result_valid(result, X), "c-step must return valid location, covariance, determinant, support, and distances")
def fast_mcd_c_step(
    X: NDArray[np.float64],
    n_support: int,
    *,
    random_state: RandomStateLike = None,
    remaining_iterations: int = 30,
    initial_location: NDArray[np.float64] | None = None,
    initial_covariance: NDArray[np.float64] | None = None,
) -> CStepResult:
    """Run sklearn's deterministic FastMCD c-step loop from a random or supplied start."""
    X_values = np.asarray(X, dtype=np.float64)
    n_samples = X_values.shape[0]
    dist = np.full(n_samples, np.inf, dtype=np.float64)

    if initial_location is None:
        support_indices = fast_mcd_initial_random_support_indices(
            n_samples,
            int(n_support),
            random_state=random_state,
        )
    else:
        support_indices = fast_mcd_support_indices_from_estimates(
            X_values,
            np.asarray(initial_location, dtype=np.float64),
            np.asarray(initial_covariance, dtype=np.float64),
            n_support=int(n_support),
        )

    location, covariance, det = fast_mcd_support_statistics(X_values, support_indices)
    if np.isinf(det):
        precision = linalg.pinvh(covariance)

    previous_det = np.inf
    while det < previous_det and remaining_iterations > 0 and not np.isinf(det):
        previous_location = location
        previous_covariance = covariance
        previous_det = det
        previous_support_indices = support_indices

        precision = linalg.pinvh(covariance)
        centered = X_values - location
        dist = np.asarray(np.sum(np.dot(centered, precision) * centered, axis=1), dtype=np.float64)
        support_indices = np.asarray(np.argpartition(dist, int(n_support) - 1)[: int(n_support)], dtype=np.int64)
        location, covariance, det = fast_mcd_support_statistics(X_values, support_indices)
        remaining_iterations -= 1

    previous_dist = dist
    dist = np.asarray(np.sum(np.dot(X_values - location, precision) * (X_values - location), axis=1), dtype=np.float64)

    if np.isinf(det):
        results = location, covariance, det, support_indices, dist
    elif np.allclose(det, previous_det):
        results = location, covariance, det, support_indices, dist
    elif det > previous_det:
        warnings.warn(
            "Determinant has increased; this should not happen: "
            "log(det) > log(previous_det) (%.15f > %.15f). "
            "You may want to try with a higher value of "
            "support_fraction (current value: %.3f)."
            % (det, previous_det, int(n_support) / X_values.shape[0]),
            RuntimeWarning,
        )
        results = (
            previous_location,
            previous_covariance,
            previous_det,
            previous_support_indices,
            previous_dist,
        )
    elif remaining_iterations == 0:
        results = location, covariance, det, support_indices, dist
    else:
        results = location, covariance, det, support_indices, dist

    location, covariance, det, support_indices, dist = results
    support = np.bincount(np.asarray(support_indices, dtype=np.int64), minlength=X_values.shape[0]).astype(bool)
    return (
        np.asarray(location, dtype=np.float64),
        np.asarray(covariance, dtype=np.float64),
        float(det),
        np.asarray(support, dtype=np.bool_),
        np.asarray(dist, dtype=np.float64),
    )
