"""Ghost witnesses for MinCovDet fit-bookkeeping helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 2 or cols < 1:
        raise ValueError(f"{name} must have at least two rows and one column")
    return rows, cols


def _check_square(values: AbstractArray, name: str) -> int:
    rows, cols = _check_matrix(values, name)
    if rows != cols:
        raise ValueError(f"{name} must be square")
    return rows


def witness_mincovdet_full_rank_warning_required(
    X: AbstractArray,
    *,
    tol: float = 1e-8,
) -> bool:
    """Describe MinCovDet's full-rank warning predicate."""
    _check_matrix(X, "X")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    return False


def witness_mincovdet_assume_centered_raw_location(
    n_features: int,
) -> AbstractArray:
    """Describe MinCovDet's zero raw-location vector for assume-centered fitting."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_mincovdet_assume_centered_raw_covariance(
    X: AbstractArray,
    raw_support: AbstractArray,
) -> AbstractArray:
    """Describe raw covariance recomputation from the supported rows."""
    _, n_features = _check_matrix(X, "X")
    if len(raw_support.shape) != 1 or int(raw_support.shape[0]) != int(X.shape[0]):
        raise ValueError("raw_support must be a sample-length vector")
    return AbstractArray(shape=(n_features, n_features), dtype="float64")


def witness_mincovdet_assume_centered_raw_distances(
    X: AbstractArray,
    raw_covariance: AbstractArray,
) -> AbstractArray:
    """Describe assume-centered raw squared distances."""
    n_samples, n_features = _check_matrix(X, "X")
    if _check_square(raw_covariance, "raw_covariance") != n_features:
        raise ValueError("raw_covariance must match X feature dimensions")
    return AbstractArray(shape=(n_samples,), dtype="float64")
