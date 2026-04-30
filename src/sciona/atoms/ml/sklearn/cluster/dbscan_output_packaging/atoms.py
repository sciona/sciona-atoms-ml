"""DBSCAN output-packaging helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dbscan_core_sample_indices,
    witness_dbscan_sparse_core_components,
)


def _core_sample_mask_valid(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.uint8)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all((array == 0) | (array == 1)))


def _core_indices_valid(result: object, core_samples: object) -> bool:
    values = np.asarray(result, dtype=np.intp)
    mask = np.asarray(core_samples, dtype=np.uint8)
    expected = np.where(mask)[0].astype(np.intp)
    return bool(values.ndim == 1 and np.array_equal(values, expected))


def _nonempty_sparse_matrix(value: object) -> bool:
    return bool(
        sp.issparse(value)
        and value.ndim == 2
        and value.shape[0] >= 1
        and value.shape[1] >= 1
        and np.all(np.isfinite(value.data))
    )


def _sparse_core_indices_valid(core_sample_indices: object, n_samples: int) -> bool:
    try:
        values = np.asarray(core_sample_indices, dtype=np.intp)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and np.all(values >= 0) and np.all(values < n_samples))


def _sparse_components_valid(result: object, X: object, core_sample_indices: object) -> bool:
    if not sp.issparse(result):
        return False
    expected = X[np.asarray(core_sample_indices, dtype=np.intp)].copy()
    return bool(
        result.shape == expected.shape
        and result.format == expected.format
        and np.array_equal(result.toarray(), expected.toarray())
    )


@register_atom(witness_dbscan_core_sample_indices)
@icontract.require(lambda core_samples: _core_sample_mask_valid(core_samples), "core_samples must be a nonempty 1D uint8-like mask of zeros and ones")
@icontract.ensure(lambda result, core_samples: _core_indices_valid(result, core_samples), "result must equal np.where(core_samples)[0]")
def dbscan_core_sample_indices(core_samples: NDArray[np.uint8]) -> NDArray[np.intp]:
    """Extract DBSCAN core-sample indices from the uint8 core mask."""
    return np.asarray(np.where(np.asarray(core_samples, dtype=np.uint8))[0], dtype=np.intp)


@register_atom(witness_dbscan_sparse_core_components)
@icontract.require(lambda X: _nonempty_sparse_matrix(X), "X must be a nonempty finite sparse matrix")
@icontract.require(lambda X, core_sample_indices: _sparse_core_indices_valid(core_sample_indices, X.shape[0]), "core_sample_indices must be a valid 1D index vector into X")
@icontract.ensure(lambda result, X, core_sample_indices: _sparse_components_valid(result, X, core_sample_indices), "result must be the copied sparse core-sample rows")
def dbscan_sparse_core_components(
    X: sp.spmatrix,
    core_sample_indices: NDArray[np.intp],
) -> sp.spmatrix:
    """Copy sparse DBSCAN core-sample rows selected by `core_sample_indices_`."""
    return X[np.asarray(core_sample_indices, dtype=np.intp)].copy()
