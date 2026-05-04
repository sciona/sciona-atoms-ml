"""GraphicalLasso fit-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.atoms.ml.sklearn.covariance import empirical_covariance
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_graphical_lasso_fit_costs,
    witness_graphical_lasso_fit_covariance,
    witness_graphical_lasso_fit_empirical_covariance,
    witness_graphical_lasso_fit_location,
    witness_graphical_lasso_fit_n_iter,
    witness_graphical_lasso_fit_precision,
    witness_graphical_lasso_fit_return_self,
    witness_graphical_lasso_fit_use_precomputed_covariance,
)

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]
CostHistory = list[tuple[float, float]]


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 2 and np.all(np.isfinite(array)))


def _finite_square_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 2 and array.shape[0] == array.shape[1] and np.all(np.isfinite(array)))


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _bool_scalar(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _finite_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
    )


def _costs_valid(values: object) -> bool:
    if not isinstance(values, list) or len(values) < 1:
        return False
    for item in values:
        if not (isinstance(item, tuple) and len(item) == 2):
            return False
        if not (_finite_scalar(item[0]) and _finite_scalar(item[1])):
            return False
    return True


def _same_shape_matrix(result: object, source: object) -> bool:
    return bool(_finite_square_matrix(result) and _finite_square_matrix(source) and np.asarray(result).shape == np.asarray(source).shape)


def _location_valid(result: object, X: object) -> bool:
    return bool(_finite_vector(result) and _finite_matrix(X) and np.asarray(result, dtype=np.float64).shape == (np.asarray(X, dtype=np.float64).shape[1],))


@register_atom(witness_graphical_lasso_fit_use_precomputed_covariance)
@icontract.require(lambda covariance_mode: _nonempty_string(covariance_mode), "covariance_mode must be a nonempty string")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def graphical_lasso_fit_use_precomputed_covariance(
    covariance_mode: str,
) -> bool:
    """Decide whether GraphicalLasso.fit uses the precomputed covariance branch."""
    return covariance_mode == "precomputed"


@register_atom(witness_graphical_lasso_fit_empirical_covariance)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite matrix with at least two samples and two features")
@icontract.require(lambda covariance_mode: _nonempty_string(covariance_mode), "covariance_mode must be a nonempty string")
@icontract.require(lambda assume_centered: _bool_scalar(assume_centered), "assume_centered must be boolean")
@icontract.ensure(lambda result: _finite_square_matrix(result), "empirical covariance must be a finite square matrix")
@icontract.ensure(lambda result, X: _same_shape_matrix(result, np.zeros((np.asarray(X).shape[1], np.asarray(X).shape[1]), dtype=np.float64)), "empirical covariance must be feature-by-feature")
def graphical_lasso_fit_empirical_covariance(
    X: Matrix,
    covariance_mode: str,
    *,
    assume_centered: bool,
) -> Matrix:
    """Resolve GraphicalLasso.fit's covariance input before the deferred sparse solver."""
    values = np.asarray(X, dtype=np.float64)
    if covariance_mode == "precomputed":
        return values.copy()
    return np.asarray(empirical_covariance(values, assume_centered=assume_centered), dtype=np.float64)


@register_atom(witness_graphical_lasso_fit_location)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite matrix with at least two samples and two features")
@icontract.require(lambda covariance_mode: _nonempty_string(covariance_mode), "covariance_mode must be a nonempty string")
@icontract.require(lambda assume_centered: _bool_scalar(assume_centered), "assume_centered must be boolean")
@icontract.ensure(lambda result, X: _location_valid(result, X), "location must be a finite feature-length vector")
def graphical_lasso_fit_location(
    X: Matrix,
    covariance_mode: str,
    *,
    assume_centered: bool,
) -> Vector:
    """Resolve GraphicalLasso.location_ before the deferred sparse solver."""
    values = np.asarray(X, dtype=np.float64)
    if covariance_mode == "precomputed" or assume_centered:
        return np.zeros(values.shape[1], dtype=np.float64)
    return np.asarray(values.mean(axis=0), dtype=np.float64)


@register_atom(witness_graphical_lasso_fit_covariance)
@icontract.require(lambda covariance: _finite_square_matrix(covariance), "covariance must be a finite square matrix with at least two features")
@icontract.ensure(lambda result, covariance: np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(covariance, dtype=np.float64)), "result must expose covariance_ unchanged")
def graphical_lasso_fit_covariance(
    covariance: Matrix,
) -> Matrix:
    """Expose GraphicalLasso.covariance_ after deferred sparse refit."""
    return np.asarray(covariance, dtype=np.float64)


@register_atom(witness_graphical_lasso_fit_precision)
@icontract.require(lambda precision: _finite_square_matrix(precision), "precision must be a finite square matrix with at least two features")
@icontract.ensure(lambda result, precision: np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(precision, dtype=np.float64)), "result must expose precision_ unchanged")
def graphical_lasso_fit_precision(
    precision: Matrix,
) -> Matrix:
    """Expose GraphicalLasso.precision_ after deferred sparse refit."""
    return np.asarray(precision, dtype=np.float64)


@register_atom(witness_graphical_lasso_fit_costs)
@icontract.require(lambda costs: _costs_valid(costs), "costs must be a nonempty list of finite (cost, dual_gap) pairs")
@icontract.ensure(lambda result, costs: isinstance(result, list) and result == costs, "result must expose costs_ unchanged")
def graphical_lasso_fit_costs(
    costs: CostHistory,
) -> CostHistory:
    """Expose GraphicalLasso.costs_ from the already-computed sparse refit."""
    return [(float(cost), float(dual_gap)) for cost, dual_gap in costs]


@register_atom(witness_graphical_lasso_fit_n_iter)
@icontract.require(lambda n_iter: _positive_int(n_iter), "n_iter must be a positive integer")
@icontract.ensure(lambda result, n_iter: _positive_int(result) and int(result) == int(n_iter), "result must expose n_iter_ unchanged")
def graphical_lasso_fit_n_iter(
    n_iter: int,
) -> int:
    """Expose GraphicalLasso.n_iter_ from the already-computed sparse refit."""
    return int(n_iter)


@register_atom(witness_graphical_lasso_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(lambda result, estimator_token: result == estimator_token, "fit return value must preserve the estimator token")
def graphical_lasso_fit_return_self(
    estimator_token: str,
) -> str:
    """Model GraphicalLasso.fit returning self after fitted-state assignment."""
    return estimator_token
