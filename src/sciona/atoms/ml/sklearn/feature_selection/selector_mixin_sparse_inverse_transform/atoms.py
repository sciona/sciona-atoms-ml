"""SelectorMixin sparse inverse-transform helper adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp

from sciona.ghost.registry import register_atom

from .witnesses import witness_selector_inverse_transform_csc


def _support_mask_valid(support_mask: object) -> bool:
    values = np.asarray(support_mask)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and values.dtype == np.bool_)


def _selected_sparse_valid(X_selected: object, support_mask: object) -> bool:
    if not (sp.issparse(X_selected) and _support_mask_valid(support_mask)):
        return False
    matrix = X_selected
    mask = np.asarray(support_mask, dtype=np.bool_)
    return bool(
        matrix.ndim == 2
        and matrix.shape[0] >= 1
        and matrix.shape[1] == int(np.sum(mask))
        and np.all(np.isfinite(matrix.data))
    )


def _inverse_sparse_result_valid(result: object, X_selected: object, support_mask: object) -> bool:
    if not sp.isspmatrix_csc(result):
        return False
    selected = X_selected.tocsc()
    mask = np.asarray(support_mask, dtype=np.bool_)
    expected = np.zeros(mask.shape[0], dtype=np.int64)
    expected[mask] = np.diff(selected.indptr)
    values = np.diff(result.indptr)
    return bool(
        result.shape == (selected.shape[0], mask.shape[0])
        and np.all(np.isfinite(result.data))
        and np.array_equal(values, expected)
        and np.array_equal(result.data, selected.data)
        and np.array_equal(result.indices, selected.indices)
    )


@register_atom(witness_selector_inverse_transform_csc)
@icontract.require(
    lambda X_selected, support_mask: _selected_sparse_valid(X_selected, support_mask),
    "X_selected must be a finite sparse matrix whose column count matches the number of selected features",
)
@icontract.ensure(
    lambda result, X_selected, support_mask: _inverse_sparse_result_valid(result, X_selected, support_mask),
    "inverse transform must expand sparse selected columns back into CSC format with zero-filled dropped columns",
)
def selector_inverse_transform_csc(
    X_selected: sp.spmatrix,
    support_mask: np.ndarray,
) -> sp.csc_matrix:
    """Reinsert zero-filled dropped columns into a sparse selected-feature matrix as CSC output."""
    selected = X_selected.tocsc()
    mask = np.asarray(support_mask, dtype=np.bool_)
    selected_column_nonzeros = np.diff(selected.indptr)
    restored_column_nonzeros = np.zeros(mask.shape[0], dtype=np.int64)
    restored_column_nonzeros[mask] = selected_column_nonzeros
    indptr = np.concatenate([[0], np.cumsum(restored_column_nonzeros, dtype=np.int64)])
    return sp.csc_matrix(
        (selected.data, selected.indices, indptr),
        shape=(selected.shape[0], mask.shape[0]),
        dtype=selected.dtype,
    )
