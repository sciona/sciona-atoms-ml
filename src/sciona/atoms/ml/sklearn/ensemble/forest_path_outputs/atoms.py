"""Forest apply and decision-path output helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_forest_apply_leaf_matrix,
    witness_forest_decision_path_csr,
    witness_forest_decision_path_node_ptr,
)

LeafVector = NDArray[np.integer]
LeafVectorTuple = tuple[LeafVector, ...]
SparseMatrixTuple = tuple[sp.spmatrix, ...]


def _leaf_vector_valid(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.issubdtype(array.dtype, np.integer))


def _leaf_vectors_aligned(leaf_vectors: object) -> bool:
    if not isinstance(leaf_vectors, tuple) or len(leaf_vectors) < 1:
        return False
    widths = []
    for values in leaf_vectors:
        if not _leaf_vector_valid(values):
            return False
        widths.append(int(np.asarray(values).shape[0]))
    return len(set(widths)) == 1


def _leaf_matrix_valid(result: object, leaf_vectors: LeafVectorTuple) -> bool:
    try:
        matrix = np.asarray(result)
    except (TypeError, ValueError):
        return False
    n_estimators = len(leaf_vectors)
    n_samples = int(np.asarray(leaf_vectors[0]).shape[0])
    return bool(
        matrix.ndim == 2
        and matrix.shape == (n_samples, n_estimators)
        and np.issubdtype(matrix.dtype, np.integer)
    )


def _sparse_indicator_valid(values: object) -> bool:
    return bool(sp.issparse(values) and values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1)


def _sparse_indicators_aligned(indicators: object) -> bool:
    if not isinstance(indicators, tuple) or len(indicators) < 1:
        return False
    sample_counts = []
    for block in indicators:
        if not _sparse_indicator_valid(block):
            return False
        sample_counts.append(int(block.shape[0]))
    return len(set(sample_counts)) == 1


def _node_ptr_valid(result: object, indicators: SparseMatrixTuple) -> bool:
    values = np.asarray(result)
    return bool(
        values.ndim == 1
        and values.shape[0] == len(indicators) + 1
        and np.issubdtype(values.dtype, np.integer)
        and np.array_equal(values, np.cumsum([0, *[int(block.shape[1]) for block in indicators]], dtype=np.int64))
    )


def _stacked_indicator_valid(result: object, indicators: SparseMatrixTuple) -> bool:
    if not sp.issparse(result):
        return False
    width = sum(int(block.shape[1]) for block in indicators)
    return bool(result.ndim == 2 and result.shape == (int(indicators[0].shape[0]), width) and result.format == "csr")


@register_atom(witness_forest_apply_leaf_matrix)
@icontract.require(
    lambda leaf_vectors: _leaf_vectors_aligned(leaf_vectors),
    "leaf_vectors must be a nonempty tuple of aligned 1D integer vectors",
)
@icontract.ensure(
    lambda result, leaf_vectors: _leaf_matrix_valid(result, leaf_vectors),
    "leaf matrix must transpose estimator-major leaf vectors into sample-major layout",
)
def forest_apply_leaf_matrix(
    leaf_vectors: LeafVectorTuple,
) -> NDArray[np.int64]:
    """Transpose one leaf-index vector per tree into sklearn's apply output matrix."""
    return np.asarray(leaf_vectors, dtype=np.int64).T


@register_atom(witness_forest_decision_path_node_ptr)
@icontract.require(
    lambda indicators: _sparse_indicators_aligned(indicators),
    "indicators must be a nonempty tuple of aligned 2D sparse indicator matrices",
)
@icontract.ensure(
    lambda result, indicators: _node_ptr_valid(result, indicators),
    "n_nodes_ptr must be the cumulative column-width offsets for the indicator blocks",
)
def forest_decision_path_node_ptr(
    indicators: SparseMatrixTuple,
) -> NDArray[np.int64]:
    """Return sklearn's cumulative decision-path column offsets for per-tree indicator blocks."""
    return np.cumsum([0, *[int(block.shape[1]) for block in indicators]], dtype=np.int64)


@register_atom(witness_forest_decision_path_csr)
@icontract.require(
    lambda indicators: _sparse_indicators_aligned(indicators),
    "indicators must be a nonempty tuple of aligned 2D sparse indicator matrices",
)
@icontract.ensure(
    lambda result, indicators: _stacked_indicator_valid(result, indicators),
    "stacked decision-path indicator must be CSR with concatenated estimator columns",
)
def forest_decision_path_csr(
    indicators: SparseMatrixTuple,
) -> sp.csr_matrix:
    """Horizontally stack per-tree decision-path indicators the way sklearn returns them."""
    return sp.hstack(indicators).tocsr()
