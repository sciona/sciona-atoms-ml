"""Sklearn tree path-API atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_apply_leaf_indices,
    witness_tree_decision_path_indicator,
)


def _leaf_index_vector(values: object) -> bool:
    array = np.asarray(values)
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.issubdtype(array.dtype, np.integer)
        and np.all(array >= 0)
    )


def _decision_path_indicator(values: object) -> bool:
    if not issparse(values):
        return False
    matrix = csr_matrix(values)
    return bool(
        matrix.ndim == 2
        and matrix.shape[0] >= 1
        and matrix.shape[1] >= 1
        and np.issubdtype(matrix.indices.dtype, np.integer)
        and np.issubdtype(matrix.indptr.dtype, np.integer)
    )


@register_atom(witness_tree_apply_leaf_indices)
@icontract.require(
    lambda leaf_indices: _leaf_index_vector(leaf_indices),
    "leaf_indices must be a nonempty one-dimensional nonnegative integer vector",
)
@icontract.ensure(
    lambda result, leaf_indices: _leaf_index_vector(result)
    and np.array_equal(np.asarray(result), np.asarray(leaf_indices)),
    "apply leaf indices must preserve the supplied vector",
)
def tree_apply_leaf_indices(
    leaf_indices: NDArray[np.integer],
) -> NDArray[np.integer]:
    """Return BaseDecisionTree.apply's final leaf-index vector."""
    return np.asarray(leaf_indices)


@register_atom(witness_tree_decision_path_indicator)
@icontract.require(
    lambda indicator: _decision_path_indicator(indicator),
    "indicator must be a nonempty sparse decision-path matrix",
)
@icontract.ensure(
    lambda result, indicator: _decision_path_indicator(result)
    and csr_matrix(result).shape == csr_matrix(indicator).shape,
    "decision-path indicator must preserve sparse shape and structure kind",
)
def tree_decision_path_indicator(
    indicator: csr_matrix,
) -> csr_matrix:
    """Return BaseDecisionTree.decision_path's final sparse indicator matrix."""
    return csr_matrix(indicator)

