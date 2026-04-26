"""Ghost witnesses for dense sklearn.metrics.pairwise kernel helpers."""

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


def witness_pairwise_default_gamma(n_features: int, gamma: float | None = None) -> float:
    """Describe sklearn's default gamma fallback of 1 / n_features."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if gamma is not None and gamma < 0.0:
        raise ValueError("gamma must be non-negative when provided")
    return 1.0 / float(n_features) if gamma is None else float(gamma)


def witness_pairwise_linear_kernel(
    X: AbstractArray,
    Y: AbstractArray | None = None,
) -> AbstractArray:
    """Describe a dense linear-kernel Gram matrix."""
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_pairwise_polynomial_kernel(
    X: AbstractArray,
    Y: AbstractArray | None = None,
    *,
    degree: float = 3.0,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> AbstractArray:
    """Describe a dense polynomial-kernel Gram matrix."""
    if degree < 1.0:
        raise ValueError("degree must be at least 1")
    if gamma is not None and gamma < 0.0:
        raise ValueError("gamma must be non-negative when provided")
    del coef0
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_pairwise_laplacian_kernel(
    X: AbstractArray,
    Y: AbstractArray | None = None,
    *,
    gamma: float | None = None,
) -> AbstractArray:
    """Describe a dense Laplacian-kernel Gram matrix."""
    if gamma is not None and gamma <= 0.0:
        raise ValueError("gamma must be strictly positive when provided")
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_pairwise_sigmoid_kernel(
    X: AbstractArray,
    Y: AbstractArray | None = None,
    *,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> AbstractArray:
    """Describe a dense sigmoid-kernel Gram matrix."""
    if gamma is not None and gamma < 0.0:
        raise ValueError("gamma must be non-negative when provided")
    del coef0
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_pairwise_cosine_similarity(
    X: AbstractArray,
    Y: AbstractArray | None = None,
) -> AbstractArray:
    """Describe a dense cosine-similarity matrix."""
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")
