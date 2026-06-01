from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy import sparse


def test_quantile_sparse_lp_matrix_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_sparse_lp_matrix_shell import (
        quantile_highs_sparse_a_eq,
        quantile_highs_intercept_column,
        quantile_highs_sparse_identity,
    )

    assert callable(quantile_highs_sparse_identity)
    assert callable(quantile_highs_intercept_column)
    assert callable(quantile_highs_sparse_a_eq)


def test_quantile_highs_sparse_identity_matches_source_block() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_sparse_lp_matrix_shell import quantile_highs_sparse_identity

    result = quantile_highs_sparse_identity(3, np.float32)

    assert sparse.isspmatrix_csc(result)
    assert result.dtype == np.dtype(np.float32)
    assert np.allclose(result.toarray(), np.eye(3, dtype=np.float32))


def test_quantile_highs_intercept_column_matches_source_block() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_sparse_lp_matrix_shell import quantile_highs_intercept_column

    result = quantile_highs_intercept_column(4, np.float64)

    assert sparse.isspmatrix_csc(result)
    assert result.dtype == np.dtype(np.float64)
    assert np.allclose(result.toarray(), np.ones((4, 1), dtype=np.float64))


def test_quantile_highs_sparse_a_eq_matches_source_with_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_sparse_lp_matrix_shell import quantile_highs_sparse_a_eq

    X = sparse.csr_matrix(np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]], dtype=np.float64))

    result = quantile_highs_sparse_a_eq(X, fit_intercept=True)

    eye = sparse.eye(X.shape[0], dtype=X.dtype, format="csc")
    ones = sparse.csc_matrix(np.ones(shape=(X.shape[0], 1), dtype=X.dtype))
    expected = sparse.hstack([ones, X, -ones, -X, eye, -eye], format="csc")
    assert sparse.isspmatrix_csc(result)
    assert result.shape == (3, 12)
    assert np.allclose(result.toarray(), expected.toarray())


def test_quantile_highs_sparse_a_eq_matches_source_without_intercept_for_dense_x() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_sparse_lp_matrix_shell import quantile_highs_sparse_a_eq

    X = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=np.float32)

    result = quantile_highs_sparse_a_eq(X, fit_intercept=False)

    eye = sparse.eye(X.shape[0], dtype=X.dtype, format="csc")
    expected = sparse.hstack([X, -X, eye, -eye], format="csc")
    assert sparse.isspmatrix_csc(result)
    assert result.dtype == np.dtype(np.float32)
    assert result.shape == (2, 8)
    assert np.allclose(result.toarray(), expected.toarray())


def test_quantile_sparse_lp_matrix_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_sparse_lp_matrix_shell import (
        quantile_highs_sparse_a_eq,
        quantile_highs_intercept_column,
        quantile_highs_sparse_identity,
    )

    with pytest.raises(ViolationError):
        quantile_highs_sparse_identity(0, np.float64)

    with pytest.raises(ViolationError):
        quantile_highs_intercept_column(2, object())

    with pytest.raises(ViolationError):
        quantile_highs_sparse_a_eq(np.ones(3, dtype=np.float64), fit_intercept=True)

    with pytest.raises(ViolationError):
        quantile_highs_sparse_a_eq(sparse.csr_matrix([[1.0, np.nan]]), fit_intercept=True)

    with pytest.raises(ViolationError):
        quantile_highs_sparse_a_eq(np.ones((2, 2), dtype=np.float64), fit_intercept=1)
