"""Sklearn QuantileRegressor sparse LP matrix atoms."""

from __future__ import annotations

import icontract
import numpy as np
from scipy import sparse

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_quantile_highs_sparse_a_eq,
    witness_quantile_highs_intercept_column,
    witness_quantile_highs_sparse_identity,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _dtype_valid(dtype: object) -> bool:
    try:
        np.dtype(dtype)
    except TypeError:
        return False
    return True


def _bool_valid(value: bool) -> bool:
    return isinstance(value, bool)


def _matrix_valid(value: object) -> bool:
    if sparse.issparse(value):
        return bool(len(value.shape) == 2 and value.shape[0] >= 1 and value.shape[1] >= 1 and np.all(np.isfinite(value.data)))
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _csc_shape(result: object, n_rows: int, n_cols: int) -> bool:
    return bool(sparse.isspmatrix_csc(result) and result.shape == (n_rows, n_cols))


def _constraint_shape(result: object, X: object, fit_intercept: bool) -> bool:
    n_rows, n_features = X.shape
    n_params = n_features + int(fit_intercept)
    return _csc_shape(result, n_rows, 2 * n_params + 2 * n_rows)


@register_atom(witness_quantile_highs_sparse_identity)
@icontract.require(lambda n_rows: _positive_int(n_rows), "n_rows must be positive")
@icontract.require(lambda dtype: _dtype_valid(dtype), "dtype must be a valid numpy dtype")
@icontract.ensure(lambda result, n_rows: _csc_shape(result, n_rows, n_rows), "identity block must be a CSC square matrix")
def quantile_highs_sparse_identity(n_rows: int, dtype: object) -> sparse.csc_matrix:
    """Return the CSC identity block used by QuantileRegressor for HiGHS."""
    return sparse.eye(n_rows, dtype=np.dtype(dtype), format="csc")


@register_atom(witness_quantile_highs_intercept_column)
@icontract.require(lambda n_rows: _positive_int(n_rows), "n_rows must be positive")
@icontract.require(lambda dtype: _dtype_valid(dtype), "dtype must be a valid numpy dtype")
@icontract.ensure(lambda result, n_rows: _csc_shape(result, n_rows, 1), "intercept block must be a CSC column")
def quantile_highs_intercept_column(n_rows: int, dtype: object) -> sparse.csc_matrix:
    """Return the CSC ones column used by QuantileRegressor for HiGHS."""
    return sparse.csc_matrix(np.ones(shape=(n_rows, 1), dtype=np.dtype(dtype)))


@register_atom(witness_quantile_highs_sparse_a_eq)
@icontract.require(lambda X: _matrix_valid(X), "X must be a finite 2D dense or sparse matrix")
@icontract.require(lambda fit_intercept: _bool_valid(fit_intercept), "fit_intercept must be boolean")
@icontract.ensure(
    lambda result, X, fit_intercept: _constraint_shape(result, X, fit_intercept),
    "constraint matrix must match sklearn HiGHS CSC layout",
)
def quantile_highs_sparse_a_eq(X: object, *, fit_intercept: bool) -> sparse.csc_matrix:
    """Return the sparse equality-constraint matrix used for HiGHS solvers."""
    n_rows = X.shape[0]
    eye = quantile_highs_sparse_identity(n_rows, X.dtype)
    if fit_intercept:
        ones = quantile_highs_intercept_column(n_rows, X.dtype)
        return sparse.hstack([ones, X, -ones, -X, eye, -eye], format="csc")
    return sparse.hstack([X, -X, eye, -eye], format="csc")
