"""Ghost witnesses for sklearn Birch sparse-iteration helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_birch_sparse_row_bounds(indptr: AbstractArray, row_index: int) -> tuple[int, int]:
    """Describe CSR row bounds for one Birch sparse row."""
    if len(indptr.shape) != 1:
        raise ValueError("indptr must be 1D")
    if int(indptr.shape[0]) < 2:
        raise ValueError("indptr must contain at least one row")
    if row_index < 0 or row_index >= int(indptr.shape[0]) - 1:
        raise ValueError("row_index out of range")
    return 0, 0


def witness_birch_sparse_dense_row(
    n_features: int,
    nonzero_indices: AbstractArray,
    nonzero_values: AbstractArray,
) -> AbstractArray:
    """Describe one densified CSR row for Birch sparse iteration."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if len(nonzero_indices.shape) != 1 or len(nonzero_values.shape) != 1:
        raise ValueError("nonzero inputs must be 1D")
    if int(nonzero_indices.shape[0]) != int(nonzero_values.shape[0]):
        raise ValueError("nonzero arrays must have matching lengths")
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_birch_sparse_dense_rows(X_indptr: AbstractArray, X_indices: AbstractArray, X_data: AbstractArray, n_features: int) -> AbstractArray:
    """Describe stacked densified CSR rows for Birch sparse iteration."""
    if len(X_indptr.shape) != 1 or len(X_indices.shape) != 1 or len(X_data.shape) != 1:
        raise ValueError("CSR arrays must be 1D")
    if int(X_indptr.shape[0]) < 2:
        raise ValueError("X_indptr must contain at least one row")
    if int(X_indices.shape[0]) != int(X_data.shape[0]):
        raise ValueError("X_indices and X_data must have matching lengths")
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return AbstractArray(shape=(int(X_indptr.shape[0]) - 1, n_features), dtype="float64")
