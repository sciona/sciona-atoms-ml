"""Birch sparse-iteration atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_birch_sparse_dense_row,
    witness_birch_sparse_dense_rows,
    witness_birch_sparse_row_bounds,
)

IndexVector = NDArray[np.int64] | list[int]
FloatVector = NDArray[np.float64] | list[float]


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _csr_indptr_valid(indptr: object) -> bool:
    values = np.asarray(indptr)
    return bool(
        values.ndim == 1
        and values.shape[0] >= 2
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values[:-1] <= values[1:])
    )


def _csr_indices_valid(indices: object, n_features: int) -> bool:
    values = np.asarray(indices)
    return bool(
        _positive_int(n_features)
        and values.ndim == 1
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_features)
    )


def _csr_data_valid(data: object) -> bool:
    try:
        values = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and np.all(np.isfinite(values)))


def _row_index_valid(row_index: object, indptr: object) -> bool:
    return bool(
        isinstance(row_index, int)
        and not isinstance(row_index, bool)
        and _csr_indptr_valid(indptr)
        and 0 <= row_index < np.asarray(indptr).shape[0] - 1
    )


def _row_bounds_valid(result: object, row_index: int, indptr: object) -> bool:
    if not (isinstance(result, tuple) and len(result) == 2):
        return False
    startptr, endptr = result
    values = np.asarray(indptr, dtype=np.int64)
    return bool(
        isinstance(startptr, int)
        and isinstance(endptr, int)
        and startptr == int(values[row_index])
        and endptr == int(values[row_index + 1])
        and 0 <= startptr <= endptr
    )


def _matching_nonzero_vectors(nonzero_indices: object, nonzero_values: object, n_features: int) -> bool:
    try:
        values = np.asarray(nonzero_values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    indices = np.asarray(nonzero_indices)
    return bool(
        _csr_indices_valid(indices, n_features)
        and values.ndim == 1
        and indices.ndim == 1
        and indices.shape[0] == values.shape[0]
        and np.all(np.isfinite(values))
    )


def _dense_row_valid(result: object, n_features: int) -> bool:
    try:
        values = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.shape == (n_features,) and np.all(np.isfinite(values)))


def _csr_triplet_valid(X_indptr: object, X_indices: object, X_data: object, n_features: int) -> bool:
    if not (_csr_indptr_valid(X_indptr) and _csr_indices_valid(X_indices, n_features) and _csr_data_valid(X_data)):
        return False
    indptr = np.asarray(X_indptr, dtype=np.int64)
    indices = np.asarray(X_indices)
    data = np.asarray(X_data, dtype=np.float64)
    return bool(indptr[-1] == indices.shape[0] == data.shape[0])


def _dense_rows_valid(result: object, X_indptr: object, n_features: int) -> bool:
    try:
        values = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.shape == (np.asarray(X_indptr).shape[0] - 1, n_features)
        and np.all(np.isfinite(values))
    )


@register_atom(witness_birch_sparse_row_bounds)
@icontract.require(lambda indptr: _csr_indptr_valid(indptr), "indptr must be a valid nondecreasing CSR pointer vector")
@icontract.require(lambda row_index, indptr: _row_index_valid(row_index, indptr), "row_index must address one CSR row")
@icontract.ensure(lambda result, row_index, indptr: _row_bounds_valid(result, row_index, indptr), "row bounds must match the CSR pointer vector")
def birch_sparse_row_bounds(indptr: IndexVector, row_index: int) -> tuple[int, int]:
    """Return Birch's CSR start and end pointers for one sparse row."""
    values = np.asarray(indptr, dtype=np.int64)
    return int(values[row_index]), int(values[row_index + 1])


@register_atom(witness_birch_sparse_dense_row)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be positive")
@icontract.require(lambda nonzero_indices, nonzero_values, n_features: _matching_nonzero_vectors(nonzero_indices, nonzero_values, n_features), "nonzero_indices and nonzero_values must be matching finite CSR row slices")
@icontract.ensure(lambda result, n_features: _dense_row_valid(result, n_features), "dense row must match the feature count")
def birch_sparse_dense_row(
    n_features: int,
    nonzero_indices: IndexVector,
    nonzero_values: FloatVector,
) -> NDArray[np.float64]:
    """Return Birch's densified sparse row from one CSR row slice."""
    row = np.zeros(int(n_features), dtype=np.float64)
    indices = np.asarray(nonzero_indices, dtype=np.int64)
    values = np.asarray(nonzero_values, dtype=np.float64)
    row[indices] = values
    return row


@register_atom(witness_birch_sparse_dense_rows)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be positive")
@icontract.require(lambda X_indptr, X_indices, X_data, n_features: _csr_triplet_valid(X_indptr, X_indices, X_data, n_features), "CSR arrays must be compatible and finite")
@icontract.ensure(lambda result, X_indptr, n_features: _dense_rows_valid(result, X_indptr, n_features), "dense rows must match Birch sparse iteration output shape")
def birch_sparse_dense_rows(
    X_indptr: IndexVector,
    X_indices: IndexVector,
    X_data: FloatVector,
    n_features: int,
) -> NDArray[np.float64]:
    """Return Birch's stacked densified rows from CSR components."""
    indptr = np.asarray(X_indptr, dtype=np.int64)
    indices = np.asarray(X_indices, dtype=np.int64)
    data = np.asarray(X_data, dtype=np.float64)
    rows = np.zeros((indptr.shape[0] - 1, int(n_features)), dtype=np.float64)
    for row_index in range(rows.shape[0]):
        startptr, endptr = birch_sparse_row_bounds(indptr, row_index)
        rows[row_index] = birch_sparse_dense_row(
            int(n_features),
            indices[startptr:endptr],
            data[startptr:endptr],
        )
    return rows
