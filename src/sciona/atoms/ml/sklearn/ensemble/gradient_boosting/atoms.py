"""Deterministic gradient-boosting helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gradient_boosting_huber_delta,
    witness_gradient_boosting_safe_divide,
)

def _finite_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))

def _finite_vector(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))

def _sample_weight_valid(sample_weight: NDArray[np.float64], y_true: NDArray[np.float64]) -> bool:
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
        targets = np.asarray(y_true, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        weights.ndim == 1
        and targets.ndim == 1
        and weights.shape == targets.shape
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.sum(weights) > 0.0
    )

def _quantile_valid(quantile: float) -> bool:
    return bool(_finite_scalar(quantile) and 0.0 < float(quantile) < 1.0)

def _shapes_align(y_true: NDArray[np.float64], raw_prediction: NDArray[np.float64]) -> bool:
    y_values = np.asarray(y_true)
    raw_values = np.asarray(raw_prediction)
    return bool(y_values.ndim == 1 and raw_values.ndim in {1, 2} and raw_values.shape[0] == y_values.shape[0] and (raw_values.ndim == 1 or raw_values.shape[1] == 1))

def _safe_divide_result_valid(result: float, numerator: float, denominator: float) -> bool:
    if abs(float(denominator)) < 1e-150:
        return bool(float(result) == 0.0)
    return bool(np.isfinite(float(result)) == np.isfinite(float(numerator) / float(denominator)) and float(result) == float(numerator) / float(denominator))

def _huber_delta_result_valid(result: float) -> bool:
    return bool(np.isfinite(float(result)) and float(result) >= 0.0)

@register_atom(witness_gradient_boosting_safe_divide)
@icontract.require(lambda numerator: _finite_scalar(numerator), "numerator must be finite")
@icontract.require(lambda denominator: _finite_scalar(denominator), "denominator must be finite")
@icontract.ensure(lambda result, numerator, denominator: _safe_divide_result_valid(result, numerator, denominator), "result must follow sklearn's tiny-denominator guard and otherwise equal float division")
def gradient_boosting_safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Divide with sklearn's tiny-denominator guard used in gradient-boosting line searches."""
    if abs(float(denominator)) < 1e-150:
        return 0.0
    return float(float(numerator) / float(denominator))

@register_atom(witness_gradient_boosting_huber_delta)
@icontract.require(lambda y_true: _finite_vector(y_true), "y_true must be a finite vector")
@icontract.require(lambda raw_prediction: _finite_vector(np.asarray(raw_prediction, dtype=np.float64).reshape(-1)), "raw_prediction must be finite")
@icontract.require(lambda y_true, raw_prediction: _shapes_align(y_true, raw_prediction), "raw_prediction must align with y_true and be one-dimensional or a single-column matrix")
@icontract.require(lambda sample_weight, y_true: _sample_weight_valid(sample_weight, y_true), "sample_weight must be a finite nonnegative vector matching y_true with positive total weight")
@icontract.require(lambda quantile: _quantile_valid(quantile), "quantile must be strictly between 0 and 1")
@icontract.ensure(lambda result: _huber_delta_result_valid(result), "delta must be a finite nonnegative percentile of absolute residuals")
def gradient_boosting_huber_delta(
    y_true: NDArray[np.float64],
    raw_prediction: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    *,
    quantile: float = 0.9,
) -> float:
    from sklearn.utils.stats import _weighted_percentile
    """Compute the weighted Huber delta that sklearn gradient boosting stores for a regression stage."""
    target = np.asarray(y_true, dtype=np.float64)
    raw = np.asarray(raw_prediction, dtype=np.float64).reshape(-1)
    weights = np.asarray(sample_weight, dtype=np.float64)
    absolute_error = np.abs(target - raw)
    delta = _weighted_percentile(absolute_error, weights, 100.0 * float(quantile))
    return float(delta)
