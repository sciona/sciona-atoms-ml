"""Dense quantile-regression LP atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_quantile_dense_lp_problem,
    witness_quantile_nonzero_weight_mask,
    witness_quantile_solution_to_params,
)


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_vector(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _quantile_valid(quantile: float) -> bool:
    return bool(_finite_scalar(quantile) and 0.0 < float(quantile) < 1.0)


def _alpha_valid(alpha: float) -> bool:
    return bool(_finite_scalar(alpha) and float(alpha) >= 0.0)


def _bool_valid(value: bool) -> bool:
    return isinstance(value, bool)


def _feature_count_valid(n_features: int) -> bool:
    return bool(isinstance(n_features, int) and not isinstance(n_features, bool) and n_features >= 1)


def _sample_counts_match(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X)
    values_y = np.asarray(y)
    return bool(values_x.ndim == 2 and values_y.ndim == 1 and values_x.shape[0] == values_y.shape[0])


def _weights_match(sample_weight: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    weights = np.asarray(sample_weight)
    samples = np.asarray(X)
    return bool(weights.ndim == 1 and samples.ndim == 2 and weights.shape[0] == samples.shape[0])


def _weights_valid(sample_weight: NDArray[np.float64]) -> bool:
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and weights.shape[0] >= 1 and np.all(np.isfinite(weights)) and np.all(weights >= 0.0) and np.sum(weights) > 0.0)


def _solution_valid(solution: NDArray[np.float64], n_features: int, fit_intercept: bool) -> bool:
    try:
        values = np.asarray(solution, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if not _feature_count_valid(n_features) or not _bool_valid(fit_intercept):
        return False
    n_params = int(n_features) + int(fit_intercept)
    return bool(values.ndim == 1 and values.shape[0] >= 2 * n_params and np.all(np.isfinite(values)))


def _mask_result_valid(result: NDArray[np.bool_], sample_weight: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    weights = np.asarray(sample_weight)
    return bool(values.dtype == np.bool_ and values.shape == weights.shape and np.any(values))


def _lp_problem_result_valid(
    result: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    X: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    fit_intercept: bool,
) -> bool:
    c, a_eq, b_eq = result
    c_values = np.asarray(c, dtype=np.float64)
    a_values = np.asarray(a_eq, dtype=np.float64)
    b_values = np.asarray(b_eq, dtype=np.float64)
    mask = np.asarray(sample_weight, dtype=np.float64) != 0.0
    n_rows = int(np.sum(mask))
    n_features = np.asarray(X).shape[1]
    n_params = n_features + int(fit_intercept)
    n_columns = 2 * n_params + 2 * n_rows
    return bool(
        c_values.shape == (n_columns,)
        and a_values.shape == (n_rows, n_columns)
        and b_values.shape == (n_rows,)
        and np.all(np.isfinite(c_values))
        and np.all(np.isfinite(a_values))
        and np.all(np.isfinite(b_values))
    )


def _params_result_valid(result: tuple[NDArray[np.float64], float], n_features: int) -> bool:
    coef, intercept = result
    coef_values = np.asarray(coef, dtype=np.float64)
    return bool(coef_values.shape == (int(n_features),) and np.all(np.isfinite(coef_values)) and isinstance(intercept, float) and np.isfinite(intercept))


@register_atom(witness_quantile_nonzero_weight_mask)
@icontract.require(lambda sample_weight: _weights_valid(sample_weight), "sample_weight must be nonnegative with positive sum")
@icontract.ensure(lambda result, sample_weight: _mask_result_valid(result, sample_weight), "mask must align with sample_weight and keep rows")
def quantile_nonzero_weight_mask(sample_weight: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Identify samples with nonzero quantile-regression weights."""
    return np.asarray(np.asarray(sample_weight, dtype=np.float64) != 0.0, dtype=np.bool_)


@register_atom(witness_quantile_dense_lp_problem)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda y: _finite_vector(y), "y must be a finite vector")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "y must match X samples")
@icontract.require(lambda sample_weight: _weights_valid(sample_weight), "sample_weight must be nonnegative with positive sum")
@icontract.require(lambda sample_weight, X: _weights_match(sample_weight, X), "sample_weight must match X samples")
@icontract.require(lambda quantile: _quantile_valid(quantile), "quantile must be between zero and one")
@icontract.require(lambda alpha: _alpha_valid(alpha), "alpha must be nonnegative")
@icontract.require(lambda fit_intercept: _bool_valid(fit_intercept), "fit_intercept must be boolean")
@icontract.ensure(lambda result, X, sample_weight, fit_intercept: _lp_problem_result_valid(result, X, sample_weight, fit_intercept), "LP arrays must have compatible finite shapes")
def quantile_dense_lp_problem(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    *,
    quantile: float,
    alpha: float,
    fit_intercept: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build dense quantile-regression LP objective and equality arrays."""
    samples = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    mask = quantile_nonzero_weight_mask(weights)
    samples = samples[mask]
    target = target[mask]
    weights = weights[mask]

    n_rows, n_features = samples.shape
    n_params = n_features + int(fit_intercept)
    scaled_alpha = float(np.sum(weights)) * float(alpha)
    c = np.concatenate(
        [
            np.full(2 * n_params, fill_value=scaled_alpha, dtype=np.float64),
            weights * float(quantile),
            weights * (1.0 - float(quantile)),
        ]
    )
    if fit_intercept:
        c[0] = 0.0
        c[n_params] = 0.0

    eye = np.eye(n_rows, dtype=np.float64)
    if fit_intercept:
        ones = np.ones((n_rows, 1), dtype=np.float64)
        a_eq = np.concatenate([ones, samples, -ones, -samples, eye, -eye], axis=1)
    else:
        a_eq = np.concatenate([samples, -samples, eye, -eye], axis=1)
    return np.asarray(c, dtype=np.float64), np.asarray(a_eq, dtype=np.float64), np.asarray(target, dtype=np.float64)


@register_atom(witness_quantile_solution_to_params)
@icontract.require(lambda n_features: _feature_count_valid(n_features), "n_features must be a positive integer")
@icontract.require(lambda fit_intercept: _bool_valid(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda solution, n_features, fit_intercept: _solution_valid(solution, n_features, fit_intercept), "solution must contain finite parameter slacks")
@icontract.ensure(lambda result, n_features: _params_result_valid(result, n_features), "coefficient vector and intercept must be finite")
def quantile_solution_to_params(
    solution: NDArray[np.float64],
    n_features: int,
    *,
    fit_intercept: bool = True,
) -> tuple[NDArray[np.float64], float]:
    """Convert quantile LP parameter slacks to coefficients and intercept."""
    values = np.asarray(solution, dtype=np.float64)
    n_params = int(n_features) + int(fit_intercept)
    params = values[:n_params] - values[n_params : 2 * n_params]
    if fit_intercept:
        return np.asarray(params[1:], dtype=np.float64), float(params[0])
    return np.asarray(params, dtype=np.float64), 0.0
