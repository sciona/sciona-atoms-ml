"""Spectral biclustering normalization-dispatch atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from ..bicluster.atoms import (
    bicluster_bistochastic_normalize,
    bicluster_log_normalize,
    bicluster_scale_normalize,
)
from ..bicluster_sparse_preprocessing.atoms import (
    bicluster_sparse_bistochastic_normalize,
    bicluster_sparse_scale_normalize,
)
from .witnesses import (
    witness_bicluster_dense_normalized_data,
    witness_bicluster_sparse_normalized_data,
)


def _dense_method_valid(method: object) -> bool:
    return method in {"bistochastic", "scale", "log"}


def _sparse_method_valid(method: object) -> bool:
    return method in {"bistochastic", "scale"}


def _finite_dense_matrix(X: object) -> bool:
    try:
        array = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_sparse_matrix(X: object) -> bool:
    return bool(sp.issparse(X) and X.ndim == 2 and X.shape[0] >= 1 and X.shape[1] >= 1 and np.all(np.isfinite(X.data)))


def _dense_result_valid(result: object, X: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(X, dtype=np.float64)
    return bool(values.shape == source.shape and np.all(np.isfinite(values)))


def _sparse_result_valid(result: object, X: object) -> bool:
    return bool(sp.issparse(result) and result.shape == X.shape and np.all(np.isfinite(result.data)) and np.all(result.data >= 0.0))


@register_atom(witness_bicluster_dense_normalized_data)
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense matrix")
@icontract.require(lambda method: isinstance(method, str) and _dense_method_valid(method), "method must be one of 'bistochastic', 'scale', or 'log'")
@icontract.ensure(lambda result, X: _dense_result_valid(result, X), "result must be a finite dense matrix with the same shape as X")
def bicluster_dense_normalized_data(
    X: NDArray[np.float64],
    method: str,
) -> NDArray[np.float64]:
    """Dispatch dense spectral biclustering normalization by method."""
    if method == "bistochastic":
        return np.asarray(bicluster_bistochastic_normalize(np.asarray(X, dtype=np.float64)), dtype=np.float64)
    if method == "scale":
        normalized, _, _ = bicluster_scale_normalize(np.asarray(X, dtype=np.float64))
        return np.asarray(normalized, dtype=np.float64)
    return np.asarray(bicluster_log_normalize(np.asarray(X, dtype=np.float64)), dtype=np.float64)


@register_atom(witness_bicluster_sparse_normalized_data)
@icontract.require(lambda X: _finite_sparse_matrix(X), "X must be a finite sparse matrix")
@icontract.require(lambda method: isinstance(method, str) and _sparse_method_valid(method), "method must be 'bistochastic' or 'scale' for sparse input")
@icontract.ensure(lambda result, X: _sparse_result_valid(result, X), "result must be a finite nonnegative sparse matrix with the same shape as X")
def bicluster_sparse_normalized_data(
    X: sp.spmatrix,
    method: str,
) -> sp.spmatrix:
    """Dispatch sparse spectral biclustering normalization by method."""
    if method == "bistochastic":
        return bicluster_sparse_bistochastic_normalize(X)
    normalized, _, _ = bicluster_sparse_scale_normalize(X)
    return normalized
