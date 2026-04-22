"""Dense GLM objective atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_glm_dense_loss_gradient,
    witness_glm_linear_raw_prediction,
    witness_glm_log_link_half_loss_gradient,
)

_FAMILIES = {"poisson", "gamma", "tweedie"}


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


def _alpha_valid(alpha: float) -> bool:
    return bool(_finite_scalar(alpha) and float(alpha) >= 0.0)


def _family_valid(family: str) -> bool:
    return family in _FAMILIES


def _sample_counts_match(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X)
    values_y = np.asarray(y)
    return bool(values_x.ndim == 2 and values_y.ndim == 1 and values_x.shape[0] == values_y.shape[0])


def _feature_counts_match(X: NDArray[np.float64], coef: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X)
    values_coef = np.asarray(coef)
    return bool(values_x.ndim == 2 and values_coef.ndim == 1 and values_x.shape[1] == values_coef.shape[0])


def _vectors_align(left: NDArray[np.float64], right: NDArray[np.float64]) -> bool:
    values_left = np.asarray(left)
    values_right = np.asarray(right)
    return bool(values_left.ndim == 1 and values_right.ndim == 1 and values_left.shape == values_right.shape)


def _params_valid(params: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(params, dtype=np.float64)
        samples = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if values.ndim != 1 or samples.ndim != 2 or not np.all(np.isfinite(values)):
        return False
    return bool(values.shape[0] in {samples.shape[1], samples.shape[1] + 1})


def _sample_weight_valid(sample_weight: NDArray[np.float64] | None, n_samples: int) -> bool:
    if sample_weight is None:
        return True
    try:
        values = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] == n_samples and np.all(np.isfinite(values)) and np.all(values >= 0.0) and np.sum(values) > 0.0)


def _target_range_valid(y: NDArray[np.float64], family: str, power: float) -> bool:
    if not _family_valid(family) or not _finite_scalar(power):
        return False
    values = np.asarray(y, dtype=np.float64)
    if family == "poisson":
        return bool(np.all(values >= 0.0))
    if family == "gamma":
        return bool(np.all(values > 0.0))
    p = float(power)
    if p <= 0.0:
        return True
    if p < 2.0:
        return bool(np.all(values >= 0.0))
    return bool(np.all(values > 0.0))


def _pointwise_loss_gradient_values(
    y: NDArray[np.float64],
    raw_prediction: NDArray[np.float64],
    *,
    family: str,
    power: float,
    sample_weight: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    target = np.asarray(y, dtype=np.float64)
    raw = np.asarray(raw_prediction, dtype=np.float64)

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        if family == "poisson":
            exp_raw = np.exp(raw)
            loss = exp_raw - target * raw
            gradient = exp_raw - target
        elif family == "gamma":
            exp_neg_raw = np.exp(-raw)
            loss = raw + target * exp_neg_raw
            gradient = 1.0 - target * exp_neg_raw
        else:
            p = float(power)
            if p == 0.0:
                exp_raw = np.exp(raw)
                gradient = exp_raw * (exp_raw - target)
                loss = 0.5 * (target - exp_raw) ** 2
            elif p == 1.0:
                exp_raw = np.exp(raw)
                loss = exp_raw - target * raw
                gradient = exp_raw - target
            elif p == 2.0:
                exp_neg_raw = np.exp(-raw)
                loss = raw + target * exp_neg_raw
                gradient = 1.0 - target * exp_neg_raw
            else:
                exp1 = np.exp((1.0 - p) * raw)
                exp2 = np.exp((2.0 - p) * raw)
                loss = exp2 / (2.0 - p) - target * exp1 / (1.0 - p)
                gradient = exp2 - target * exp1

    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)
        loss = loss * weights
        gradient = gradient * weights
    return np.asarray(loss, dtype=np.float64), np.asarray(gradient, dtype=np.float64)


def _pointwise_values_finite(
    y: NDArray[np.float64],
    raw_prediction: NDArray[np.float64],
    family: str,
    power: float,
    sample_weight: NDArray[np.float64] | None,
) -> bool:
    try:
        loss, gradient = _pointwise_loss_gradient_values(
            y,
            raw_prediction,
            family=family,
            power=float(power),
            sample_weight=sample_weight,
        )
    except (FloatingPointError, TypeError, ValueError):
        return False
    return bool(loss.shape == np.asarray(y).shape and gradient.shape == np.asarray(y).shape and np.all(np.isfinite(loss)) and np.all(np.isfinite(gradient)))


def _raw_prediction_result_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    samples = np.asarray(X)
    return bool(values.shape == (samples.shape[0],) and np.all(np.isfinite(values)))


def _pointwise_result_valid(result: tuple[NDArray[np.float64], NDArray[np.float64]], y: NDArray[np.float64]) -> bool:
    loss, gradient = result
    loss_values = np.asarray(loss, dtype=np.float64)
    gradient_values = np.asarray(gradient, dtype=np.float64)
    target = np.asarray(y)
    return bool(loss_values.shape == target.shape and gradient_values.shape == target.shape and np.all(np.isfinite(loss_values)) and np.all(np.isfinite(gradient_values)))


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


@register_atom(witness_glm_linear_raw_prediction)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda coef: _finite_vector(coef), "coef must be a finite vector")
@icontract.require(lambda X, coef: _feature_counts_match(X, coef), "coef must match X features")
@icontract.require(lambda intercept: _finite_scalar(intercept), "intercept must be finite")
@icontract.ensure(lambda result, X: _raw_prediction_result_valid(result, X), "raw prediction must align with X samples")
def glm_linear_raw_prediction(
    X: NDArray[np.float64],
    coef: NDArray[np.float64],
    *,
    intercept: float = 0.0,
) -> NDArray[np.float64]:
    """Compute the GLM linear predictor from supplied coefficients."""
    samples = np.asarray(X, dtype=np.float64)
    weights = np.asarray(coef, dtype=np.float64)
    return np.asarray(samples.dot(weights) + float(intercept), dtype=np.float64)


@register_atom(witness_glm_log_link_half_loss_gradient)
@icontract.require(lambda y: _finite_vector(y), "y must be a finite vector")
@icontract.require(lambda raw_prediction: _finite_vector(raw_prediction), "raw_prediction must be finite")
@icontract.require(lambda y, raw_prediction: _vectors_align(y, raw_prediction), "raw_prediction must match y")
@icontract.require(lambda family: _family_valid(family), "family must be poisson, gamma, or tweedie")
@icontract.require(lambda power: _finite_scalar(power), "power must be finite")
@icontract.require(lambda y, family, power: _target_range_valid(y, family, power), "y must be in the family target range")
@icontract.require(lambda sample_weight, y: _sample_weight_valid(sample_weight, np.asarray(y).shape[0]), "sample_weight must be nonnegative and align with y")
@icontract.require(lambda y, raw_prediction, family, power, sample_weight: _pointwise_values_finite(y, raw_prediction, family, power, sample_weight), "loss and gradient must stay finite")
@icontract.ensure(lambda result, y: _pointwise_result_valid(result, y), "pointwise loss and gradient must align with y")
def glm_log_link_half_loss_gradient(
    y: NDArray[np.float64],
    raw_prediction: NDArray[np.float64],
    *,
    family: str,
    power: float = 1.5,
    sample_weight: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute pointwise log-link GLM half loss and raw gradients."""
    return _pointwise_loss_gradient_values(
        y,
        raw_prediction,
        family=family,
        power=float(power),
        sample_weight=sample_weight,
    )


@register_atom(witness_glm_dense_loss_gradient)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda y: _finite_vector(y), "y must be a finite vector")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "y must match X samples")
@icontract.require(lambda params, X: _params_valid(params, X), "params must contain coefficients and optional intercept")
@icontract.require(lambda family: _family_valid(family), "family must be poisson, gamma, or tweedie")
@icontract.require(lambda alpha: _alpha_valid(alpha), "alpha must be nonnegative")
@icontract.require(lambda power: _finite_scalar(power), "power must be finite")
@icontract.require(lambda y, family, power: _target_range_valid(y, family, power), "y must be in the family target range")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, np.asarray(X).shape[0]), "sample_weight must be nonnegative and align with X")
@icontract.ensure(lambda result, params: _loss_gradient_result_valid(result, params), "loss and gradient must be finite and aligned")
def glm_dense_loss_gradient(
    params: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    family: str,
    alpha: float = 0.0,
    power: float = 1.5,
    sample_weight: NDArray[np.float64] | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Compute dense GLM objective value and gradient at supplied parameters."""
    samples = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    values = np.asarray(params, dtype=np.float64)
    n_features = samples.shape[1]
    fit_intercept = values.shape[0] == n_features + 1
    coef = values[:n_features]
    intercept = float(values[-1]) if fit_intercept else 0.0
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    raw_prediction = glm_linear_raw_prediction(samples, coef, intercept=intercept)
    pointwise_loss, pointwise_gradient = glm_log_link_half_loss_gradient(
        target,
        raw_prediction,
        family=family,
        power=float(power),
        sample_weight=weights,
    )
    sw_sum = float(samples.shape[0]) if weights is None else float(np.sum(weights))
    loss = float(np.sum(pointwise_loss)) / sw_sum
    loss += 0.5 * float(alpha) * float(np.dot(coef, coef))

    scaled_pointwise_gradient = np.asarray(pointwise_gradient, dtype=np.float64) / sw_sum
    gradient = np.empty_like(values, dtype=np.float64)
    gradient[:n_features] = samples.T.dot(scaled_pointwise_gradient) + float(alpha) * coef
    if fit_intercept:
        gradient[-1] = float(np.sum(scaled_pointwise_gradient))
    return float(loss), np.asarray(gradient, dtype=np.float64)
