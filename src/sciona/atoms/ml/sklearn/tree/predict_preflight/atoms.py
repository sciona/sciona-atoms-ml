"""Sklearn tree prediction-preflight atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_predict_ensure_all_finite_mode,
    witness_tree_predict_require_sparse_int32_indices,
    witness_tree_predict_use_check_input_branch,
)


def _integer_vector(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.issubdtype(array.dtype, np.integer))


def _sparse_index_arrays_valid(indices: object, indptr: object) -> bool:
    return bool(_integer_vector(indices) and _integer_vector(indptr))


@register_atom(witness_tree_predict_use_check_input_branch)
@icontract.require(lambda check_input: isinstance(check_input, bool), "check_input must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def tree_predict_use_check_input_branch(check_input: bool) -> bool:
    """Return the `check_input` branch predicate used by `_validate_X_predict`."""
    return check_input


@register_atom(witness_tree_predict_ensure_all_finite_mode)
@icontract.require(
    lambda supports_missing_values: isinstance(supports_missing_values, bool),
    "supports_missing_values must be a boolean flag",
)
@icontract.ensure(
    lambda result, supports_missing_values: result == ("allow-nan" if supports_missing_values else True),
    "ensure_all_finite mode must match sklearn's missing-value support rule",
)
def tree_predict_ensure_all_finite_mode(
    supports_missing_values: bool,
) -> str | bool:
    """Return sklearn's ensure_all_finite mode for tree prediction validation."""
    return "allow-nan" if supports_missing_values else True


@register_atom(witness_tree_predict_require_sparse_int32_indices)
@icontract.require(
    lambda indices, indptr: _sparse_index_arrays_valid(indices, indptr),
    "indices and indptr must be nonempty 1D integer arrays",
)
@icontract.ensure(lambda result: result is True, "successful sparse index validation must return True")
def tree_predict_require_sparse_int32_indices(
    indices: NDArray[np.integer],
    indptr: NDArray[np.integer],
) -> bool:
    """Enforce sklearn's CSR sparse-index dtype requirement before tree prediction."""
    index_values = np.asarray(indices)
    indptr_values = np.asarray(indptr)
    if index_values.dtype != np.intc or indptr_values.dtype != np.intc:
        raise ValueError("No support for np.int64 index based sparse matrices")
    return True

