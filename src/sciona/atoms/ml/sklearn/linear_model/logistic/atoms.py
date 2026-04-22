"""Dense binary logistic objective atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_binary_logistic_dense_loss_gradient,
    witness_binary_logistic_half_loss_gradient,
    witness_binary_logistic_positive_probability,
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


def _alpha_valid(alpha: float) -> bool:
    return bool(_finite_scalar(alpha) and float(alpha) >= 0.0)


def _sample_counts_match(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X)
    values_y = np.asarray(y)
    return bool(values_x.ndim == 2 and values_y.ndim == 1 and values_x.shape[0] == values_y.shape[0])


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


def _target_valid(y: NDArray[np.float64]) -> bool:
    values = np.asarray(y, dtype=np.float64)
    return bool(np.all((0.0 <= values) & (values <= 1.0)))


def _sample_weight_valid(sample_weight: NDArray[np.float64] | None, n_samples: int) -> bool:
    if sample_weight is None:
        return True
    try:
        values = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] == n_samples and np.all(np.isfinite(values)) and np.all(values >= 0.0) and np.sum(values) > 0.0)


def _linear_raw_prediction(
    X: NDArray[np.float64],
    params: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    samples = np.asarray(X, dtype=np.float64)
    values = np.asarray(params, dtype=np.float64)
    n_features = samples.shape[1]
    coef = values[:n_features]
    intercept = float(values[-1]) if values.shape[0] == n_features + 1 else 0.0
    raw_prediction = np.asarray(samples.dot(coef) + intercept, dtype=np.float64)
    return coef, raw_prediction, intercept


def _pointwise_loss_gradient_values(
    y: NDArray[np.float64],
    raw_prediction: NDArray[np.float64],
    *,
    sample_weight: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    target = np.asarray(y, dtype=np.float64)
    raw = np.asarray(raw_prediction, dtype=np.float64)
    loss = np.empty_like(raw, dtype=np.float64)
    gradient = np.empty_like(raw, dtype=np.float64)

    mask_low = raw <= -37.0
    if np.any(mask_low):
        exp_raw = np.exp(raw[mask_low])
        loss[mask_low] = exp_raw - target[mask_low] * raw[mask_low]
        gradient[mask_low] = exp_raw - target[mask_low]

    mask_mid_low = (raw > -37.0) & (raw <= -2.0)
    if np.any(mask_mid_low):
        exp_raw = np.exp(raw[mask_mid_low])
        target_slice = target[mask_mid_low]
        loss[mask_mid_low] = np.log1p(exp_raw) - target_slice * raw[mask_mid_low]
        gradient[mask_mid_low] = ((1.0 - target_slice) * exp_raw - target_slice) / (1.0 + exp_raw)

    mask_mid_high = (raw > -2.0) & (raw <= 18.0)
    if np.any(mask_mid_high):
        exp_neg_raw = np.exp(-raw[mask_mid_high])
        target_slice = target[mask_mid_high]
        loss[mask_mid_high] = np.log1p(exp_neg_raw) + (1.0 - target_slice) * raw[mask_mid_high]
        gradient[mask_mid_high] = ((1.0 - target_slice) - target_slice * exp_neg_raw) / (1.0 + exp_neg_raw)

    mask_high = raw > 18.0
    if np.any(mask_high):
        exp_neg_raw = np.exp(-raw[mask_high])
        target_slice = target[mask_high]
        loss[mask_high] = exp_neg_raw + (1.0 - target_slice) * raw[mask_high]
        gradient[mask_high] = ((1.0 - target_slice) - target_slice * exp_neg_raw) / (1.0 + exp_neg_raw)

    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)
        loss = loss * weights
        gradient = gradient * weights
    return np.asarray(loss, dtype=np.float64), np.asarray(gradient, dtype=np.float64)


def _pointwise_values_finite(
    y: NDArray[np.float64],
    raw_prediction: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None,
) -> bool:
    try:
        loss, gradient = _pointwise_loss_gradient_values(y, raw_prediction, sample_weight=sample_weight)
    except (FloatingPointError, TypeError, ValueError):
        return False
    return bool(loss.shape == np.asarray(y).shape and gradient.shape == np.asarray(y).shape and np.all(np.isfinite(loss)) and np.all(np.isfinite(gradient)))


def _probability_result_valid(result: NDArray[np.float64], raw_prediction: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    raw_values = np.asarray(raw_prediction)
    return bool(values.shape == raw_values.shape and np.all(np.isfinite(values)) and np.all((0.0 <= values) & (values <= 1.0)))


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


@register_atom(witness_binary_logistic_positive_probability)
@icontract.require(lambda raw_prediction: _finite_vector(raw_prediction), "raw_prediction must be finite")
@icontract.ensure(lambda result, raw_prediction: _probability_result_valid(result, raw_prediction), "probabilities must align with raw_prediction")
def binary_logistic_positive_probability(raw_prediction: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute positive-class probabilities from binary logistic raw scores."""
    return np.asarray(expit(np.asarray(raw_prediction, dtype=np.float64)), dtype=np.float64)


@register_atom(witness_binary_logistic_half_loss_gradient)
@icontract.require(lambda y: _finite_vector(y), "y must be a finite vector")
@icontract.require(lambda raw_prediction: _finite_vector(raw_prediction), "raw_prediction must be finite")
@icontract.require(lambda y, raw_prediction: _vectors_align(y, raw_prediction), "raw_prediction must match y")
@icontract.require(lambda y: _target_valid(y), "y must be in the closed unit interval")
@icontract.require(lambda sample_weight, y: _sample_weight_valid(sample_weight, np.asarray(y).shape[0]), "sample_weight must be nonnegative and align with y")
@icontract.require(lambda y, raw_prediction, sample_weight: _pointwise_values_finite(y, raw_prediction, sample_weight), "loss and gradient must stay finite")
@icontract.ensure(lambda result, y: _pointwise_result_valid(result, y), "pointwise loss and gradient must align with y")
def binary_logistic_half_loss_gradient(
    y: NDArray[np.float64],
    raw_prediction: NDArray[np.float64],
    *,
    sample_weight: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute pointwise binary logistic half loss and raw gradients."""
    return _pointwise_loss_gradient_values(y, raw_prediction, sample_weight=sample_weight)


@register_atom(witness_binary_logistic_dense_loss_gradient)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda y: _finite_vector(y), "y must be a finite vector")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "y must match X samples")
@icontract.require(lambda params, X: _params_valid(params, X), "params must contain coefficients and optional intercept")
@icontract.require(lambda y: _target_valid(y), "y must be in the closed unit interval")
@icontract.require(lambda alpha: _alpha_valid(alpha), "alpha must be nonnegative")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, np.asarray(X).shape[0]), "sample_weight must be nonnegative and align with X")
@icontract.ensure(lambda result, params: _loss_gradient_result_valid(result, params), "loss and gradient must be finite and aligned")
def binary_logistic_dense_loss_gradient(
    params: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    alpha: float = 0.0,
    sample_weight: NDArray[np.float64] | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Compute dense binary logistic objective value and gradient at supplied parameters."""
    samples = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    values = np.asarray(params, dtype=np.float64)
    n_features = samples.shape[1]
    fit_intercept = values.shape[0] == n_features + 1
    coef, raw_prediction, _ = _linear_raw_prediction(samples, values)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    pointwise_loss, pointwise_gradient = binary_logistic_half_loss_gradient(
        target,
        raw_prediction,
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
