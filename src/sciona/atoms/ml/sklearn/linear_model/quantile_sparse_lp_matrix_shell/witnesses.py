"""Ghost witnesses for sklearn QuantileRegressor sparse LP matrix atoms."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def witness_quantile_highs_sparse_identity(n_rows: int, dtype: object) -> object:
    """Describe the CSC identity block used for HiGHS LP constraints."""
    return sparse.eye(n_rows, dtype=np.dtype(dtype), format="csc")


def witness_quantile_highs_intercept_column(n_rows: int, dtype: object) -> object:
    """Describe the CSC intercept column used for HiGHS LP constraints."""
    return sparse.csc_matrix(np.ones(shape=(n_rows, 1), dtype=np.dtype(dtype)))


def witness_quantile_highs_sparse_a_eq(X: object, fit_intercept: bool) -> object:
    """Describe the CSC equality-constraint matrix assembled for HiGHS."""
    n_rows = X.shape[0]
    eye = sparse.eye(n_rows, dtype=X.dtype, format="csc")
    if fit_intercept:
        ones = sparse.csc_matrix(np.ones(shape=(n_rows, 1), dtype=X.dtype))
        return sparse.hstack([ones, X, -ones, -X, eye, -eye], format="csc")
    return sparse.hstack([X, -X, eye, -eye], format="csc")
