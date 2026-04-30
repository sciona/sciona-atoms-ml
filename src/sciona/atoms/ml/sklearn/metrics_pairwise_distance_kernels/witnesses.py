"""Ghost witnesses for dense sklearn.metrics.pairwise distance kernels."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_xy(X: AbstractArray, Y: AbstractArray | None) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if Y is None:
        return int(X.shape[0]), int(X.shape[0])
    if len(Y.shape) != 2:
        raise ValueError("Y must be 2D")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have matching feature counts")
    return int(X.shape[0]), int(Y.shape[0])


def witness_pairwise_rbf_kernel(
    X: AbstractArray,
    Y: AbstractArray | None = None,
    *,
    gamma: float | None = None,
) -> AbstractArray:
    """Describe a dense radial-basis-function kernel matrix."""
    if gamma is not None and gamma <= 0.0:
        raise ValueError("gamma must be strictly positive when provided")
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_pairwise_additive_chi2_kernel(
    X: AbstractArray,
    Y: AbstractArray | None = None,
) -> AbstractArray:
    """Describe a dense additive chi-squared kernel matrix."""
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_pairwise_chi2_kernel(
    X: AbstractArray,
    Y: AbstractArray | None = None,
    *,
    gamma: float = 1.0,
) -> AbstractArray:
    """Describe a dense exponentiated chi-squared kernel matrix."""
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive")
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")
