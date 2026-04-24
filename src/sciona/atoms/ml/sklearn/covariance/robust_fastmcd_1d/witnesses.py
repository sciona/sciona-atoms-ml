"""Ghost witnesses for one-dimensional FastMCD helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_positive_int(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _check_vector_or_column(values: AbstractArray, name: str) -> int:
    if len(values.shape) == 1:
        rows = int(values.shape[0])
        if rows < 2:
            raise ValueError(f"{name} must contain at least two samples")
        return rows
    if len(values.shape) == 2:
        rows, cols = int(values.shape[0]), int(values.shape[1])
        if rows < 2 or cols != 1:
            raise ValueError(f"{name} must be a nonempty column matrix")
        return rows
    raise ValueError(f"{name} must be a vector or column matrix")


def witness_fast_mcd_support_count(
    n_samples: int,
    n_features: int,
    *,
    support_fraction: float | None = None,
) -> int:
    """Describe the support size used by FastMCD."""
    _check_positive_int(n_samples, "n_samples")
    _check_positive_int(n_features, "n_features")
    if support_fraction is not None and not (0.0 < support_fraction <= 1.0):
        raise ValueError("support_fraction must lie in (0, 1]")
    return 1


def witness_fast_mcd_1d_location(
    X: AbstractArray,
    n_support: int,
) -> AbstractArray:
    """Describe the one-dimensional FastMCD location output."""
    _check_positive_int(n_support, "n_support")
    _check_vector_or_column(X, "X")
    return AbstractArray(shape=(1,), dtype="float64")


def witness_fast_mcd_1d_support_mask(
    X: AbstractArray,
    location: AbstractArray,
    n_support: int,
) -> AbstractArray:
    """Describe the one-dimensional FastMCD support mask."""
    n_samples = _check_vector_or_column(X, "X")
    _check_positive_int(n_support, "n_support")
    if len(location.shape) != 1 or int(location.shape[0]) != 1:
        raise ValueError("location must have shape (1,)")
    return AbstractArray(shape=(n_samples,), dtype="bool")


def witness_fast_mcd_1d_covariance(
    X: AbstractArray,
    support_mask: AbstractArray,
) -> AbstractArray:
    """Describe the one-dimensional FastMCD covariance output."""
    _check_vector_or_column(X, "X")
    if len(support_mask.shape) != 1 or int(support_mask.shape[0]) < 2:
        raise ValueError("support_mask must be a nonempty vector")
    return AbstractArray(shape=(1, 1), dtype="float64")


def witness_fast_mcd_1d_squared_distances(
    X: AbstractArray,
    location: AbstractArray,
    covariance: AbstractArray,
) -> AbstractArray:
    """Describe the one-dimensional FastMCD squared-distance output."""
    n_samples = _check_vector_or_column(X, "X")
    if len(location.shape) != 1 or int(location.shape[0]) != 1:
        raise ValueError("location must have shape (1,)")
    if len(covariance.shape) != 2 or tuple(map(int, covariance.shape)) != (1, 1):
        raise ValueError("covariance must have shape (1, 1)")
    return AbstractArray(shape=(n_samples,), dtype="float64", min_val=0)
