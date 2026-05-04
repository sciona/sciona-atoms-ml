"""GraphicalLassoCV postfit-state atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_graphical_lasso_cv_fit_alpha,
    witness_graphical_lasso_cv_fit_costs,
    witness_graphical_lasso_cv_fit_covariance,
    witness_graphical_lasso_cv_fit_n_iter,
    witness_graphical_lasso_cv_fit_precision,
    witness_graphical_lasso_cv_fit_return_self,
)

Matrix = NDArray[np.float64]
CostHistory = list[tuple[float, float]]


def _finite_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float, np.floating, np.integer))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
    )


def _nonnegative_scalar(value: object) -> bool:
    return bool(_finite_scalar(value) and float(value) >= 0.0)


def _finite_square_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 2
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
    )


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _costs_valid(values: object) -> bool:
    if not isinstance(values, list) or len(values) < 1:
        return False
    for item in values:
        if not (isinstance(item, tuple) and len(item) == 2):
            return False
        cost, dual_gap = item
        if not (_finite_scalar(cost) and _finite_scalar(dual_gap)):
            return False
    return True


def _same_matrix(result: object, source: object) -> bool:
    return bool(
        _finite_square_matrix(result)
        and _finite_square_matrix(source)
        and np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(source, dtype=np.float64))
    )


def _same_costs(result: object, source: object) -> bool:
    return bool(_costs_valid(result) and _costs_valid(source) and result == source)


@register_atom(witness_graphical_lasso_cv_fit_alpha)
@icontract.require(lambda alpha: _nonnegative_scalar(alpha), "alpha must be a finite nonnegative scalar")
@icontract.ensure(lambda result, alpha: _nonnegative_scalar(result) and float(result) == float(alpha), "result must expose alpha_ unchanged")
def graphical_lasso_cv_fit_alpha(
    alpha: float,
) -> float:
    """Expose GraphicalLassoCV.alpha_ after deferred path selection and sparse refit."""
    return float(alpha)


@register_atom(witness_graphical_lasso_cv_fit_covariance)
@icontract.require(lambda covariance: _finite_square_matrix(covariance), "covariance must be a finite square matrix with at least two features")
@icontract.ensure(lambda result, covariance: _same_matrix(result, covariance), "result must expose covariance_ unchanged")
def graphical_lasso_cv_fit_covariance(
    covariance: Matrix,
) -> Matrix:
    """Expose GraphicalLassoCV.covariance_ after deferred sparse refit."""
    return np.asarray(covariance, dtype=np.float64)


@register_atom(witness_graphical_lasso_cv_fit_precision)
@icontract.require(lambda precision: _finite_square_matrix(precision), "precision must be a finite square matrix with at least two features")
@icontract.ensure(lambda result, precision: _same_matrix(result, precision), "result must expose precision_ unchanged")
def graphical_lasso_cv_fit_precision(
    precision: Matrix,
) -> Matrix:
    """Expose GraphicalLassoCV.precision_ after deferred sparse refit."""
    return np.asarray(precision, dtype=np.float64)


@register_atom(witness_graphical_lasso_cv_fit_costs)
@icontract.require(lambda costs: _costs_valid(costs), "costs must be a nonempty list of finite (cost, dual_gap) pairs")
@icontract.ensure(lambda result, costs: _same_costs(result, costs), "result must expose costs_ unchanged")
def graphical_lasso_cv_fit_costs(
    costs: CostHistory,
) -> CostHistory:
    """Expose GraphicalLassoCV.costs_ from the already-computed sparse refit."""
    return [(float(cost), float(dual_gap)) for cost, dual_gap in costs]


@register_atom(witness_graphical_lasso_cv_fit_n_iter)
@icontract.require(lambda n_iter: _positive_int(n_iter), "n_iter must be a positive integer")
@icontract.ensure(lambda result, n_iter: _positive_int(result) and int(result) == int(n_iter), "result must expose n_iter_ unchanged")
def graphical_lasso_cv_fit_n_iter(
    n_iter: int,
) -> int:
    """Expose GraphicalLassoCV.n_iter_ from the already-computed sparse refit."""
    return int(n_iter)


@register_atom(witness_graphical_lasso_cv_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(lambda result, estimator_token: result == estimator_token, "fit return value must preserve the estimator token")
def graphical_lasso_cv_fit_return_self(
    estimator_token: str,
) -> str:
    """Model GraphicalLassoCV.fit returning self after fitted-state assignment."""
    return estimator_token
