"""Witnesses for sklearn covariance MinCovDet correction helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    length = int(values.shape[0])
    if length < 1:
        raise ValueError(f"{name} must be nonempty")
    return length


def _check_square(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be two-dimensional")
    rows = int(values.shape[0])
    cols = int(values.shape[1])
    if rows < 1 or cols < 1 or rows != cols:
        raise ValueError(f"{name} must be a nonempty square matrix")
    return rows


def witness_mincovdet_correct_covariance_guard(
    raw_covariance: AbstractArray,
    raw_support: AbstractArray,
    raw_distances: AbstractArray,
) -> bool:
    """Describe MinCovDet's raw-covariance zero guard before correction."""
    _check_square(raw_covariance, "raw_covariance")
    n_samples = _check_vector(raw_support, "raw_support")
    if _check_vector(raw_distances, "raw_distances") != n_samples:
        raise ValueError("raw_distances length must match raw_support length")
    return True


def witness_mincovdet_empirical_correction_factor(
    raw_distances: AbstractArray,
    n_features: int,
) -> float:
    """Describe MinCovDet's empirical covariance-correction factor."""
    _check_vector(raw_distances, "raw_distances")
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return 0.0


def witness_mincovdet_corrected_covariance(
    raw_covariance: AbstractArray,
    correction_factor: float,
) -> AbstractArray:
    """Describe corrected covariance after empirical MinCovDet scaling."""
    n_features = _check_square(raw_covariance, "raw_covariance")
    if correction_factor <= 0.0:
        raise ValueError("correction_factor must be positive")
    return AbstractArray(shape=(n_features, n_features), dtype="float64")


def witness_mincovdet_corrected_distances(
    raw_distances: AbstractArray,
    correction_factor: float,
) -> AbstractArray:
    """Describe corrected distances after empirical MinCovDet scaling."""
    n_samples = _check_vector(raw_distances, "raw_distances")
    if correction_factor <= 0.0:
        raise ValueError("correction_factor must be positive")
    return AbstractArray(shape=(n_samples,), dtype="float64")
