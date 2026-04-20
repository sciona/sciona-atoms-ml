"""Ghost witnesses for sklearn covariance helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_empirical_covariance(
    X: AbstractArray,
    *,
    assume_centered: bool = False,
) -> AbstractArray:
    """Describe the square covariance matrix produced from samples."""
    del assume_centered
    if len(X.shape) not in {1, 2}:
        raise ValueError("X must be 1D or 2D")
    n_features = X.shape[0] if len(X.shape) == 1 else X.shape[1]
    return AbstractArray(shape=(n_features, n_features), dtype="float64")


def witness_shrunk_covariance(
    emp_cov: AbstractArray,
    shrinkage: float = 0.1,
) -> AbstractArray:
    """Describe covariance shrinkage that preserves matrix shape."""
    if len(emp_cov.shape) < 2:
        raise ValueError("emp_cov must be at least 2D")
    if emp_cov.shape[-1] != emp_cov.shape[-2]:
        raise ValueError("last two dimensions must be square")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    return AbstractArray(shape=emp_cov.shape, dtype=emp_cov.dtype)


def witness_ledoit_wolf_shrinkage(
    X: AbstractArray,
    *,
    assume_centered: bool = False,
    block_size: int = 1000,
) -> float:
    """Describe a bounded Ledoit-Wolf shrinkage coefficient."""
    del assume_centered
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if block_size < 1:
        raise ValueError("block_size must be at least one")
    return 0.0


def witness_ledoit_wolf(
    X: AbstractArray,
    *,
    assume_centered: bool = False,
    block_size: int = 1000,
) -> tuple[AbstractArray, float]:
    """Describe Ledoit-Wolf covariance and shrinkage outputs."""
    del assume_centered
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if block_size < 1:
        raise ValueError("block_size must be at least one")
    n_features = X.shape[1]
    return AbstractArray(shape=(n_features, n_features), dtype="float64"), 0.0


def witness_oas(
    X: AbstractArray,
    *,
    assume_centered: bool = False,
) -> tuple[AbstractArray, float]:
    """Describe OAS covariance and shrinkage outputs."""
    del assume_centered
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_features = X.shape[1]
    return AbstractArray(shape=(n_features, n_features), dtype="float64"), 0.0
