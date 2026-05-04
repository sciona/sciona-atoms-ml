"""Ghost witnesses for covariance error-norm helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_square(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or rows != cols:
        raise ValueError(f"{name} must be a nonempty square matrix")
    return rows


def witness_covariance_error_matrix(
    comp_cov: AbstractArray,
    covariance: AbstractArray,
) -> AbstractArray:
    """Describe the covariance-difference matrix used by error_norm."""
    size = _check_square(comp_cov, "comp_cov")
    if _check_square(covariance, "covariance") != size:
        raise ValueError("covariance must match comp_cov")
    return AbstractArray(shape=(size, size), dtype="float64")


def witness_covariance_error_squared_norm(
    error: AbstractArray,
    norm: str = "frobenius",
) -> AbstractArray:
    """Describe error_norm's squared norm before optional scaling."""
    _check_square(error, "error")
    if norm not in {"frobenius", "spectral"}:
        raise ValueError("norm must be 'frobenius' or 'spectral'")
    return AbstractArray(shape=(), dtype="float64")


def witness_covariance_error_scaled_squared_norm(
    squared_norm: float,
    scaling: bool = True,
    n_features: int = 1,
) -> AbstractArray:
    """Describe the optionally scaled squared norm in error_norm."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    del squared_norm, scaling
    return AbstractArray(shape=(), dtype="float64")


def witness_covariance_error_result(
    squared_norm: float,
    squared: bool = True,
) -> AbstractArray:
    """Describe the final error_norm scalar after optional square root."""
    del squared_norm, squared
    return AbstractArray(shape=(), dtype="float64")
