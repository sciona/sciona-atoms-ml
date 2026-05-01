"""Ghost witnesses for sparse spectral-biclustering preprocessing atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_rows, n_cols = int(X.shape[0]), int(X.shape[1])
    if n_rows < 1 or n_cols < 1:
        raise ValueError("X must be nonempty")
    return n_rows, n_cols


def witness_bicluster_sparse_scale_normalize(
    X: AbstractArray,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe sparse scale-normalized matrix and row/column factors."""
    n_rows, n_cols = _check_matrix(X)
    return (
        AbstractArray(shape=(n_rows, n_cols), dtype="float64", min_val=0.0),
        AbstractArray(shape=(n_rows,), dtype="float64", min_val=0.0),
        AbstractArray(shape=(n_cols,), dtype="float64", min_val=0.0),
    )


def witness_bicluster_sparse_bistochastic_distance(
    current: AbstractArray,
    original: AbstractArray,
) -> float:
    """Describe sparse bistochastic stopping distance."""
    if _check_matrix(current) != _check_matrix(original):
        raise ValueError("current and original must share a shape")
    return 0.0


def witness_bicluster_sparse_bistochastic_normalize(
    X: AbstractArray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-5,
) -> AbstractArray:
    """Describe sparse bistochastic normalization output."""
    n_rows, n_cols = _check_matrix(X)
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    return AbstractArray(shape=(n_rows, n_cols), dtype="float64", min_val=0.0)
