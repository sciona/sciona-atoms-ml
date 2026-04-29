"""Multioutput chain fit-time CV update helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_chain_fit_cv_update_required,
    witness_chain_fit_dense_cv_feature_update,
    witness_chain_fit_feature_column_index,
    witness_chain_fit_sparse_cv_feature_update,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _dense_augmented_matrix_valid(X_aug: NDArray[np.float64], cv_column: NDArray[np.float64], col_idx: int) -> bool:
    matrix = np.asarray(X_aug, dtype=np.float64)
    column = np.asarray(cv_column, dtype=np.float64)
    return bool(
        matrix.ndim == 2
        and column.ndim == 1
        and matrix.shape[0] >= 1
        and matrix.shape[1] >= 1
        and column.shape[0] == matrix.shape[0]
        and 0 <= col_idx < matrix.shape[1]
        and np.all(np.isfinite(matrix))
        and np.all(np.isfinite(column))
    )


def _dense_update_result_valid(
    result: NDArray[np.float64],
    X_aug: NDArray[np.float64],
    cv_column: NDArray[np.float64],
    col_idx: int,
) -> bool:
    updated = np.asarray(result, dtype=np.float64)
    original = np.asarray(X_aug, dtype=np.float64)
    column = np.asarray(cv_column, dtype=np.float64)
    if updated.shape != original.shape or not np.all(np.isfinite(updated)):
        return False
    if not np.allclose(updated[:, col_idx], column):
        return False
    preserved = np.delete(updated, col_idx, axis=1)
    expected = np.delete(original, col_idx, axis=1)
    return bool(np.allclose(preserved, expected))


def _sparse_augmented_matrix_valid(X_aug: object, cv_column: NDArray[np.float64], col_idx: int) -> bool:
    if not sp.issparse(X_aug):
        return False
    column = np.asarray(cv_column, dtype=np.float64)
    return bool(
        column.ndim == 1
        and X_aug.shape[0] >= 1
        and X_aug.shape[1] >= 1
        and column.shape[0] == X_aug.shape[0]
        and 0 <= col_idx < X_aug.shape[1]
        and np.all(np.isfinite(column))
    )


def _sparse_update_result_valid(result: object, X_aug: object, cv_column: NDArray[np.float64], col_idx: int) -> bool:
    if not (sp.issparse(result) and sp.issparse(X_aug)):
        return False
    updated = result.toarray()
    original = X_aug.toarray()
    column = np.asarray(cv_column, dtype=np.float64)
    if updated.shape != original.shape:
        return False
    if not np.allclose(updated[:, col_idx], column):
        return False
    preserved = np.delete(updated, col_idx, axis=1)
    expected = np.delete(original, col_idx, axis=1)
    return bool(np.allclose(preserved, expected))


@register_atom(witness_chain_fit_cv_update_required)
@icontract.require(lambda chain_idx: _nonnegative_int(chain_idx), "chain_idx must be a nonnegative integer")
@icontract.require(lambda n_outputs: _positive_int(n_outputs) and n_outputs >= 2, "n_outputs must be an integer at least 2")
@icontract.require(lambda chain_idx, n_outputs: chain_idx < n_outputs, "chain_idx must be smaller than n_outputs")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def chain_fit_cv_update_required(chain_idx: int, n_outputs: int) -> bool:
    """Return whether sklearn updates a CV feature column after this chain step."""
    return bool(chain_idx < (n_outputs - 1))


@register_atom(witness_chain_fit_feature_column_index)
@icontract.require(lambda n_base_features: _positive_int(n_base_features), "n_base_features must be a positive integer")
@icontract.require(lambda chain_idx: _nonnegative_int(chain_idx), "chain_idx must be a nonnegative integer")
@icontract.ensure(lambda result, n_base_features, chain_idx: isinstance(result, int) and result == (n_base_features + chain_idx), "column index must equal n_base_features + chain_idx")
def chain_fit_feature_column_index(n_base_features: int, chain_idx: int) -> int:
    """Return the augmented feature-column index written by sklearn chain fitting."""
    return int(n_base_features + chain_idx)


@register_atom(witness_chain_fit_dense_cv_feature_update)
@icontract.require(lambda X_aug, cv_column, col_idx: _dense_augmented_matrix_valid(X_aug, cv_column, col_idx), "X_aug, cv_column, and col_idx must describe a valid dense augmented matrix update")
@icontract.ensure(lambda result, X_aug, cv_column, col_idx: _dense_update_result_valid(result, X_aug, cv_column, col_idx), "dense update must replace exactly the selected feature column")
def chain_fit_dense_cv_feature_update(
    X_aug: NDArray[np.float64],
    cv_column: NDArray[np.float64],
    col_idx: int,
) -> NDArray[np.float64]:
    """Assign one dense chain CV feature column from supplied CV outputs."""
    updated = np.asarray(X_aug, dtype=np.float64).copy()
    updated[:, col_idx] = np.asarray(cv_column, dtype=np.float64)
    return np.asarray(updated, dtype=np.float64)


@register_atom(witness_chain_fit_sparse_cv_feature_update)
@icontract.require(lambda X_aug, cv_column, col_idx: _sparse_augmented_matrix_valid(X_aug, cv_column, col_idx), "X_aug, cv_column, and col_idx must describe a valid sparse augmented matrix update")
@icontract.ensure(lambda result, X_aug, cv_column, col_idx: _sparse_update_result_valid(result, X_aug, cv_column, col_idx), "sparse update must replace exactly the selected feature column")
def chain_fit_sparse_cv_feature_update(
    X_aug: sp.spmatrix | sp.sparray,
    cv_column: NDArray[np.float64],
    col_idx: int,
) -> sp.spmatrix | sp.sparray:
    """Assign one sparse chain CV feature column from supplied CV outputs."""
    updated = X_aug.copy()
    updated[:, col_idx] = np.expand_dims(np.asarray(cv_column, dtype=np.float64), 1)
    return updated
