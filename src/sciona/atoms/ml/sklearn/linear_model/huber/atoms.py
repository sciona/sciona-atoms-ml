"""Huber-regression objective atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_huber_linear_residuals,
    witness_huber_loss_gradient,
    witness_huber_outlier_mask,
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


def _sample_counts_match(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X)
    values_y = np.asarray(y)
    return bool(values_x.ndim == 2 and values_y.ndim == 1 and values_x.shape[0] == values_y.shape[0])


def _feature_counts_match(X: NDArray[np.float64], coef: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X)
    values_coef = np.asarray(coef)
    return bool(values_x.ndim == 2 and values_coef.ndim == 1 and values_x.shape[1] == values_coef.shape[0])


def _finite_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _positive_scalar(value: float) -> bool:
    return bool(_finite_scalar(value) and float(value) > 0.0)


def _epsilon_valid(epsilon: float) -> bool:
    return bool(_finite_scalar(epsilon) and float(epsilon) >= 1.0)


def _alpha_valid(alpha: float) -> bool:
    return bool(_finite_scalar(alpha) and float(alpha) >= 0.0)


def _params_valid(params: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(params, dtype=np.float64)
        samples = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if values.ndim != 1 or samples.ndim != 2 or not np.all(np.isfinite(values)):
        return False
    return bool(values.shape[0] in {samples.shape[1] + 1, samples.shape[1] + 2} and values[-1] > 0.0)


def _sample_weight_valid(sample_weight: NDArray[np.float64] | None, n_samples: int) -> bool:
    if sample_weight is None:
        return True
    try:
        values = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] == n_samples and np.all(np.isfinite(values)) and np.all(values >= 0.0) and np.sum(values) > 0.0)


def _residual_result_valid(result: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    target = np.asarray(y)
    return bool(values.shape == target.shape and np.all(np.isfinite(values)))


def _mask_result_valid(result: NDArray[np.bool_], residuals: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    residual_values = np.asarray(residuals)
    return bool(values.dtype == np.bool_ and values.shape == residual_values.shape)


def _loss_gradient_result_valid(
    result: tuple[float, NDArray[np.float64]],
    params: NDArray[np.float64],
) -> bool:
    loss, gradient = result
    gradient_values = np.asarray(gradient, dtype=np.float64)
    params_values = np.asarray(params)
    return bool(
        isinstance(loss, float)
        and np.isfinite(loss)
        and gradient_values.shape == params_values.shape
        and np.all(np.isfinite(gradient_values))
    )


@register_atom(witness_huber_linear_residuals)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda y: _finite_vector(y), "y must be a finite vector")
@icontract.require(lambda coef: _finite_vector(coef), "coef must be a finite vector")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "y must match X samples")
@icontract.require(lambda X, coef: _feature_counts_match(X, coef), "coef must match X features")
@icontract.require(lambda intercept: _finite_scalar(intercept), "intercept must be finite")
@icontract.ensure(lambda result, y: _residual_result_valid(result, y), "residuals must align with y")
def huber_linear_residuals(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    coef: NDArray[np.float64],
    *,
    intercept: float = 0.0,
) -> NDArray[np.float64]:
    """Compute residuals from a supplied linear prediction."""
    samples = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    weights = np.asarray(coef, dtype=np.float64)
    return np.asarray(target - samples.dot(weights) - float(intercept), dtype=np.float64)


@register_atom(witness_huber_outlier_mask)
@icontract.require(lambda residuals: _finite_vector(residuals), "residuals must be finite")
@icontract.require(lambda epsilon: _epsilon_valid(epsilon), "epsilon must be at least one")
@icontract.require(lambda sigma: _positive_scalar(sigma), "sigma must be positive")
@icontract.ensure(lambda result, residuals: _mask_result_valid(result, residuals), "mask must align with residuals")
def huber_outlier_mask(
    residuals: NDArray[np.float64],
    *,
    epsilon: float,
    sigma: float,
) -> NDArray[np.bool_]:
    """Identify residuals outside the Huber quadratic region."""
    values = np.asarray(residuals, dtype=np.float64)
    return np.asarray(np.abs(values) > float(epsilon) * float(sigma), dtype=np.bool_)


@register_atom(witness_huber_loss_gradient)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda y: _finite_vector(y), "y must be a finite vector")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "y must match X samples")
@icontract.require(lambda params, X: _params_valid(params, X), "params must contain coef, optional intercept, and positive scale")
@icontract.require(lambda epsilon: _epsilon_valid(epsilon), "epsilon must be at least one")
@icontract.require(lambda alpha: _alpha_valid(alpha), "alpha must be nonnegative")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, np.asarray(X).shape[0]), "sample_weight must be nonnegative and align with X")
@icontract.ensure(lambda result, params: _loss_gradient_result_valid(result, params), "loss and gradient must be finite and aligned")
def huber_loss_gradient(
    params: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    epsilon: float,
    alpha: float,
    sample_weight: NDArray[np.float64] | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Compute Huber objective value and gradient at supplied parameters."""
    samples = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    values = np.asarray(params, dtype=np.float64)
    weights = np.ones(samples.shape[0], dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    n_features = samples.shape[1]
    fit_intercept = values.shape[0] == n_features + 2
    intercept = float(values[-2]) if fit_intercept else 0.0
    sigma = float(values[-1])
    coef = values[:n_features]
    n_samples_weighted = float(np.sum(weights))

    residuals = huber_linear_residuals(samples, target, coef, intercept=intercept)
    outliers_mask = huber_outlier_mask(residuals, epsilon=float(epsilon), sigma=sigma)
    abs_residuals = np.abs(residuals)

    outliers = abs_residuals[outliers_mask]
    outlier_weights = weights[outliers_mask]
    n_sw_outliers = float(np.sum(outlier_weights))
    outlier_loss = 2.0 * float(epsilon) * float(np.sum(outlier_weights * outliers))
    outlier_loss -= sigma * n_sw_outliers * float(epsilon) ** 2

    non_outliers = residuals[~outliers_mask]
    weighted_non_outliers = weights[~outliers_mask] * non_outliers
    squared_loss = float(np.dot(weighted_non_outliers.T, non_outliers)) / sigma

    gradient = np.zeros(values.shape[0], dtype=np.float64)
    gradient[:n_features] = 2.0 / sigma * weighted_non_outliers.dot(-samples[~outliers_mask])

    signed_outliers = np.ones_like(outliers)
    signed_outliers[residuals[outliers_mask] < 0.0] = -1.0
    signed_outlier_weights = outlier_weights * signed_outliers
    gradient[:n_features] -= 2.0 * float(epsilon) * signed_outlier_weights.dot(samples[outliers_mask])
    gradient[:n_features] += float(alpha) * 2.0 * coef

    gradient[-1] = n_samples_weighted
    gradient[-1] -= n_sw_outliers * float(epsilon) ** 2
    gradient[-1] -= squared_loss / sigma

    if fit_intercept:
        gradient[-2] = -2.0 * float(np.sum(weighted_non_outliers)) / sigma
        gradient[-2] -= 2.0 * float(epsilon) * float(np.sum(signed_outlier_weights))

    loss = n_samples_weighted * sigma + squared_loss + outlier_loss
    loss += float(alpha) * float(np.dot(coef, coef))
    return float(loss), np.asarray(gradient, dtype=np.float64)
