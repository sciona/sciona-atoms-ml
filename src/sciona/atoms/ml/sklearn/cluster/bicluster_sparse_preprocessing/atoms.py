"""Sparse spectral-biclustering preprocessing atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from scipy.linalg import norm
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_sparse_bistochastic_distance,
    witness_bicluster_sparse_bistochastic_normalize,
    witness_bicluster_sparse_scale_normalize,
)

SparseScaleNormalization = tuple[sp.spmatrix, NDArray[np.float64], NDArray[np.float64]]

def _finite_sparse_matrix(X: object) -> bool:
    return bool(
        sp.issparse(X)
        and X.ndim == 2
        and X.shape[0] >= 1
        and X.shape[1] >= 1
        and np.all(np.isfinite(X.data))
    )

def _positive_shifted_margins(X: object) -> bool:
    from sklearn.utils.extmath import make_nonnegative
    if not _finite_sparse_matrix(X):
        return False
    shifted = make_nonnegative(X.copy())
    row_sums = np.asarray(shifted.sum(axis=1), dtype=np.float64).ravel()
    col_sums = np.asarray(shifted.sum(axis=0), dtype=np.float64).ravel()
    return bool(np.all(np.isfinite(row_sums)) and np.all(np.isfinite(col_sums)) and np.all(row_sums > 0.0) and np.all(col_sums > 0.0))

def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _positive_float(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )

def _scale_result_valid(result: object, X: object) -> bool:
    if not isinstance(result, tuple) or len(result) != 3:
        return False
    normalized, row_diag, col_diag = result
    if not sp.issparse(normalized):
        return False
    rows = np.asarray(row_diag, dtype=np.float64)
    cols = np.asarray(col_diag, dtype=np.float64)
    return bool(
        normalized.shape == X.shape
        and np.all(np.isfinite(normalized.data))
        and np.all(normalized.data >= 0.0)
        and rows.shape == (X.shape[0],)
        and cols.shape == (X.shape[1],)
        and np.all(np.isfinite(rows))
        and np.all(np.isfinite(cols))
        and np.all(rows > 0.0)
        and np.all(cols > 0.0)
    )

def _distance_inputs_valid(current: object, original: object) -> bool:
    return bool(
        _finite_sparse_matrix(current)
        and _finite_sparse_matrix(original)
        and current.shape == original.shape
        and np.asarray(current.data).shape == np.asarray(original.data).shape
    )

def _nonnegative_float(result: object) -> bool:
    return bool(isinstance(result, (int, float)) and np.isfinite(float(result)) and float(result) >= 0.0)

def _normalized_sparse_valid(result: object, X: object) -> bool:
    return bool(
        sp.issparse(result)
        and result.shape == X.shape
        and np.all(np.isfinite(result.data))
        and np.all(result.data >= 0.0)
    )

@register_atom(witness_bicluster_sparse_scale_normalize)
@icontract.require(lambda X: _positive_shifted_margins(X), "X must be a finite sparse matrix whose nonnegative-shifted row and column sums are strictly positive")
@icontract.ensure(lambda result, X: _scale_result_valid(result, X), "result must contain a nonnegative sparse normalized matrix and positive row and column factors")
def bicluster_sparse_scale_normalize(
    X: sp.spmatrix,
) -> SparseScaleNormalization:
    from sklearn.utils.extmath import make_nonnegative
    """Scale sparse rows and columns for spectral biclustering preprocessing."""
    values = make_nonnegative(X.copy())
    row_diag = np.asarray(1.0 / np.sqrt(values.sum(axis=1))).squeeze()
    col_diag = np.asarray(1.0 / np.sqrt(values.sum(axis=0))).squeeze()
    row_diag = np.where(np.isnan(row_diag), 0.0, row_diag)
    col_diag = np.where(np.isnan(col_diag), 0.0, col_diag)
    n_rows, n_cols = values.shape
    r = sp.dia_matrix((row_diag, [0]), shape=(n_rows, n_rows))
    c = sp.dia_matrix((col_diag, [0]), shape=(n_cols, n_cols))
    normalized = r * values * c
    return normalized, np.asarray(row_diag, dtype=np.float64), np.asarray(col_diag, dtype=np.float64)

@register_atom(witness_bicluster_sparse_bistochastic_distance)
@icontract.require(lambda current, original: _distance_inputs_valid(current, original), "current and original must be finite sparse matrices with matching shapes and stored-value counts")
@icontract.ensure(lambda result: _nonnegative_float(result), "distance must be a finite nonnegative scalar")
def bicluster_sparse_bistochastic_distance(
    current: sp.spmatrix,
    original: sp.spmatrix,
) -> float:
    """Compute sklearn's sparse bistochastic stopping distance from stored values."""
    current_data = np.asarray(current.data, dtype=np.float64)
    original_data = np.asarray(original.data, dtype=np.float64)
    return float(norm(current_data - original_data))

@register_atom(witness_bicluster_sparse_bistochastic_normalize)
@icontract.require(lambda X: _positive_shifted_margins(X), "X must be a finite sparse matrix whose nonnegative-shifted row and column sums are strictly positive")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be a positive integer")
@icontract.require(lambda tol: _positive_float(tol), "tol must be positive")
@icontract.ensure(lambda result, X: _normalized_sparse_valid(result, X), "result must be a nonnegative sparse matrix with the same shape as X")
def bicluster_sparse_bistochastic_normalize(
    X: sp.spmatrix,
    *,
    max_iter: int = 1000,
    tol: float = 1e-5,
) -> sp.spmatrix:
    from sklearn.utils.extmath import make_nonnegative
    """Iteratively normalize sparse biclustering data toward balanced margins."""
    original = make_nonnegative(X.copy())
    scaled = original
    for _ in range(int(max_iter)):
        scaled_new, _, _ = bicluster_sparse_scale_normalize(scaled)
        dist = bicluster_sparse_bistochastic_distance(scaled, original)
        scaled = scaled_new
        if dist < float(tol):
            break
    return scaled
