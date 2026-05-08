"""Sklearn coordinate-descent path-residual projection atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot

from sciona.ghost.registry import register_atom

from .witnesses import witness_cd_path_residuals_project_test_coefs


def _finite_numeric_array(value: object, ndim: int | None = None) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(
        np.issubdtype(array.dtype, np.number)
        and np.all(np.isfinite(array))
        and (ndim is None or array.ndim == ndim)
    )


def _finite_numeric_matrix(value: object) -> bool:
    if sparse.issparse(value):
        return bool(value.ndim == 2 and np.issubdtype(value.dtype, np.number) and np.all(np.isfinite(value.data)))
    return _finite_numeric_array(value, ndim=2)


def _projection_result_valid(result: object, X_test: object, coefs: object) -> bool:
    if not (_finite_numeric_matrix(X_test) and _finite_numeric_array(coefs, ndim=3)):
        return False
    if X_test.shape[1] != np.asarray(coefs).shape[1]:
        return False
    try:
        result_values = np.asarray(result)
        expected = np.asarray(safe_sparse_dot(X_test, coefs))
    except (TypeError, ValueError):
        return False
    coef_values = np.asarray(coefs)
    return bool(
        np.issubdtype(result_values.dtype, np.number)
        and np.all(np.isfinite(result_values))
        and result_values.shape == (X_test.shape[0], coef_values.shape[0], coef_values.shape[2])
        and np.allclose(result_values, expected)
    )


@register_atom(witness_cd_path_residuals_project_test_coefs)
@icontract.require(lambda X_test: _finite_numeric_matrix(X_test), "X_test must be a finite numeric rank-2 matrix")
@icontract.require(lambda coefs: _finite_numeric_array(coefs, ndim=3), "coefs must be a finite numeric rank-3 tensor")
@icontract.require(
    lambda X_test, coefs: X_test.shape[1] == np.asarray(coefs).shape[1],
    "X_test feature count must match coefficient feature axis",
)
@icontract.ensure(
    lambda result, X_test, coefs: _projection_result_valid(result, X_test, coefs),
    "projection must match safe_sparse_dot(X_test, coefs) with sample-output-alpha shape",
)
def cd_path_residuals_project_test_coefs(
    X_test: NDArray[np.floating] | sparse.spmatrix,
    coefs: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return sklearn's held-out coefficient-path projection in _path_residuals."""
    return safe_sparse_dot(X_test, coefs)
