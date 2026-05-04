"""GaussianProcessRegressor postfit-attribute atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_regression_fit_L,
    witness_gp_regression_fit_alpha,
    witness_gp_regression_fit_log_marginal_likelihood_value,
    witness_gp_regression_fit_return_self,
)

Matrix = NDArray[np.float64]
VectorOrMatrix = NDArray[np.float64]


def _finite_square_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
    )


def _lower_cholesky_factor(values: object) -> bool:
    if not _finite_square_matrix(values):
        return False
    factor = np.asarray(values, dtype=np.float64)
    return bool(np.allclose(factor, np.tril(factor)) and np.all(np.diag(factor) > 0.0))


def _finite_vector_or_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim in {1, 2} and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
    )


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


@register_atom(witness_gp_regression_fit_L)
@icontract.require(lambda L: _lower_cholesky_factor(L), "L must be a finite lower-triangular Cholesky factor with positive diagonal")
@icontract.ensure(
    lambda result, L: _lower_cholesky_factor(result) and np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(L, dtype=np.float64)),
    "result must expose L_ unchanged",
)
def gp_regression_fit_L(
    L: Matrix,
) -> Matrix:
    """Expose GaussianProcessRegressor.L_ after the already-computed Cholesky step."""
    return np.asarray(L, dtype=np.float64)


@register_atom(witness_gp_regression_fit_alpha)
@icontract.require(lambda alpha: _finite_vector_or_matrix(alpha), "alpha must be a finite nonempty vector or matrix")
@icontract.ensure(
    lambda result, alpha: _finite_vector_or_matrix(result) and np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(alpha, dtype=np.float64)),
    "result must expose alpha_ unchanged",
)
def gp_regression_fit_alpha(
    alpha: VectorOrMatrix,
) -> VectorOrMatrix:
    """Expose GaussianProcessRegressor.alpha_ after the already-computed dual solve."""
    return np.asarray(alpha, dtype=np.float64)


@register_atom(witness_gp_regression_fit_log_marginal_likelihood_value)
@icontract.require(lambda value: _finite_scalar(value), "value must be a finite scalar")
@icontract.ensure(lambda result, value: _finite_scalar(result) and float(result) == float(value), "result must expose log_marginal_likelihood_value_ unchanged")
def gp_regression_fit_log_marginal_likelihood_value(
    value: float,
) -> float:
    """Expose GaussianProcessRegressor.log_marginal_likelihood_value_ after optimizer or direct evaluation."""
    return float(value)


@register_atom(witness_gp_regression_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(lambda result, estimator_token: result == estimator_token, "fit return value must preserve the estimator token")
def gp_regression_fit_return_self(
    estimator_token: str,
) -> str:
    """Model GaussianProcessRegressor.fit returning self after fitted-state assignment."""
    return estimator_token
