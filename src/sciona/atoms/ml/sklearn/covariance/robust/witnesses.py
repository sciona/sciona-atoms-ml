"""Ghost witnesses for robust-covariance helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be a vector")
    length = int(values.shape[0])
    if length < 1:
        raise ValueError(f"{name} must be nonempty")
    return length


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def _check_square(values: AbstractArray, name: str) -> int:
    rows, cols = _check_matrix(values, name)
    if rows != cols:
        raise ValueError(f"{name} must be square")
    return rows


def witness_mcd_consistency_factor(n_features: int, alpha: float) -> float:
    """Describe the scalar MCD consistency factor."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie between zero and one")
    return 0.0


def witness_mcd_correct_covariance(
    raw_covariance: AbstractArray,
    dist: AbstractArray,
    n_support: int,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe MCD covariance and distance correction."""
    n_features = _check_square(raw_covariance, "raw_covariance")
    n_samples = _check_vector(dist, "dist")
    if n_support < 1 or n_support > n_samples:
        raise ValueError("n_support must fit the sample count")
    return AbstractArray(shape=(n_features, n_features), dtype="float64"), AbstractArray(shape=(n_samples,), dtype="float64")


def witness_mcd_reweight_support_mask(
    dist: AbstractArray,
    n_features: int,
    *,
    quantile_threshold: float = 0.025,
) -> AbstractArray:
    """Describe MCD chi-square reweight support selection."""
    n_samples = _check_vector(dist, "dist")
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if not 0.0 < quantile_threshold < 1.0:
        raise ValueError("quantile_threshold must lie between zero and one")
    return AbstractArray(shape=(n_samples,), dtype="bool")


def witness_mcd_reweighted_location_covariance(
    data: AbstractArray,
    support_mask: AbstractArray,
    *,
    assume_centered: bool = False,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe MCD reweighted location, covariance, and support outputs."""
    del assume_centered
    n_samples, n_features = _check_matrix(data, "data")
    if _check_vector(support_mask, "support_mask") != n_samples:
        raise ValueError("support mask length must match data rows")
    return (
        AbstractArray(shape=(n_features,), dtype="float64"),
        AbstractArray(shape=(n_features, n_features), dtype="float64"),
        AbstractArray(shape=(n_samples,), dtype="bool"),
    )


def witness_mcd_squared_mahalanobis(
    X: AbstractArray,
    location: AbstractArray,
    precision: AbstractArray,
) -> AbstractArray:
    """Describe squared Mahalanobis distances from robust covariance state."""
    n_samples, n_features = _check_matrix(X, "X")
    if _check_vector(location, "location") != n_features:
        raise ValueError("location length must match feature count")
    if _check_square(precision, "precision") != n_features:
        raise ValueError("precision shape must match feature count")
    return AbstractArray(shape=(n_samples,), dtype="float64")
