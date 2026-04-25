"""SelectorMixin sparse transform helper adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp

from sciona.ghost.registry import register_atom

from .witnesses import witness_selector_transform_sparse


def _support_mask_valid(support_mask: object) -> bool:
    values = np.asarray(support_mask)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and values.dtype == np.bool_)


def _csr_inputs_valid(X: object, support_mask: object) -> bool:
    if not (sp.isspmatrix_csr(X) and _support_mask_valid(support_mask)):
        return False
    mask = np.asarray(support_mask, dtype=np.bool_)
    return bool(X.ndim == 2 and X.shape[0] >= 1 and X.shape[1] == mask.shape[0] and np.all(np.isfinite(X.data)))


def _sparse_transform_result_valid(result: object, X: object, support_mask: object) -> bool:
    matrix = X
    mask = np.asarray(support_mask, dtype=np.bool_)
    if mask.any():
        return bool(
            sp.isspmatrix_csr(result)
            and result.shape == (matrix.shape[0], int(np.sum(mask)))
            and np.array_equal(result.toarray(), matrix[:, mask].toarray())
        )
    values = np.asarray(result, dtype=matrix.dtype)
    return bool(values.shape == (matrix.shape[0], 0) and values.ndim == 2)


@register_atom(witness_selector_transform_sparse)
@icontract.require(
    lambda X, support_mask: _csr_inputs_valid(X, support_mask),
    "X must be a finite CSR matrix whose column count matches support_mask",
)
@icontract.ensure(
    lambda result, X, support_mask: _sparse_transform_result_valid(result, X, support_mask),
    "transform result must match sklearn's sparse SelectorMixin._transform behavior",
)
def selector_transform_sparse(
    X: sp.csr_matrix,
    support_mask: np.ndarray,
) -> sp.csr_matrix | np.ndarray:
    """Reduce a CSR feature matrix to selected columns, or return sklearn's empty dense fallback."""
    mask = np.asarray(support_mask, dtype=np.bool_)
    if not mask.any():
        return np.empty((X.shape[0], 0), dtype=X.dtype)
    return X[:, mask]
