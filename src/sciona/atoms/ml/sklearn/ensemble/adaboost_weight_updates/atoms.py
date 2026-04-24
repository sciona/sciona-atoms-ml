"""AdaBoost training-stage weight update helpers adapted from scikit-learn."""

from __future__ import annotations

from typing import Literal

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_adaboost_classifier_estimator_error,
    witness_adaboost_classifier_estimator_weight,
    witness_adaboost_classifier_sample_weight_update,
    witness_adaboost_regressor_beta,
    witness_adaboost_regressor_estimator_error,
    witness_adaboost_regressor_estimator_weight,
    witness_adaboost_regressor_loss_vector,
    witness_adaboost_regressor_sample_weight_update,
)

AdaBoostR2Loss = Literal["linear", "square", "exponential"]


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _binary_or_bool_vector(values: object) -> bool:
    array = np.asarray(values)
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and (array.dtype == np.bool_ or np.issubdtype(array.dtype, np.integer))
        and np.all((array == 0) | (array == 1))
    )


def _weight_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.all(np.isfinite(array))
        and np.all(array >= 0.0)
        and np.sum(array) > 0.0
    )


def _normalized_weight_vector(values: object) -> bool:
    return _weight_vector(values) and bool(np.isclose(np.sum(np.asarray(values, dtype=np.float64)), 1.0))


def _matching_classifier_inputs(incorrect: object, sample_weight: object) -> bool:
    return bool(
        _binary_or_bool_vector(incorrect)
        and _weight_vector(sample_weight)
        and np.asarray(incorrect).shape == np.asarray(sample_weight, dtype=np.float64).shape
    )


def _matching_regressor_inputs(loss_vector: object, sample_weight: object) -> bool:
    try:
        loss_values = np.asarray(loss_vector, dtype=np.float64)
        weight_values = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        loss_values.ndim == 1
        and weight_values.ndim == 1
        and loss_values.shape == weight_values.shape
        and loss_values.shape[0] >= 1
        and np.all(np.isfinite(loss_values))
        and np.all((0.0 <= loss_values) & (loss_values <= 1.0))
        and _normalized_weight_vector(sample_weight)
    )


def _finite_nonnegative_vector(result: NDArray[np.float64], expected_shape: tuple[int, ...]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _bounded_loss_vector(result: NDArray[np.float64], absolute_errors: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == np.asarray(absolute_errors, dtype=np.float64).shape
        and np.all(np.isfinite(values))
        and np.all((0.0 <= values) & (values <= 1.0))
    )


@register_atom(witness_adaboost_classifier_estimator_error)
@icontract.require(
    lambda incorrect, sample_weight: _matching_classifier_inputs(incorrect, sample_weight),
    "incorrect must be a one-dimensional boolean or 0/1 vector aligned with a finite nonnegative sample-weight vector",
)
@icontract.ensure(
    lambda result: np.isfinite(result) and 0.0 <= float(result) <= 1.0,
    "classifier stage error must be finite and lie in [0, 1]",
)
def adaboost_classifier_estimator_error(
    incorrect: NDArray[np.bool_] | NDArray[np.int64],
    sample_weight: NDArray[np.float64],
) -> float:
    """Compute AdaBoostClassifier's weighted stage error from a mistake mask."""
    return float(
        np.mean(
            np.average(
                np.asarray(incorrect, dtype=np.float64),
                weights=np.asarray(sample_weight, dtype=np.float64),
                axis=0,
            )
        )
    )


@register_atom(witness_adaboost_classifier_estimator_weight)
@icontract.require(
    lambda estimator_error, learning_rate, n_classes: (
        np.isfinite(estimator_error)
        and np.isfinite(learning_rate)
        and learning_rate > 0.0
        and isinstance(n_classes, int)
        and not isinstance(n_classes, bool)
        and n_classes >= 2
        and 0.0 < estimator_error < 1.0 - (1.0 / n_classes)
    ),
    "estimator_error must be in the valid SAMME range, learning_rate must be positive, and n_classes must be at least 2",
)
@icontract.ensure(lambda result: np.isfinite(result), "classifier estimator weight must be finite")
def adaboost_classifier_estimator_weight(
    estimator_error: float,
    learning_rate: float,
    n_classes: int,
) -> float:
    """Compute AdaBoostClassifier's SAMME stage weight."""
    return float(
        learning_rate
        * (np.log((1.0 - estimator_error) / estimator_error) + np.log(float(n_classes - 1)))
    )


@register_atom(witness_adaboost_classifier_sample_weight_update)
@icontract.require(
    lambda sample_weight, incorrect, estimator_weight: (
        _matching_classifier_inputs(incorrect, sample_weight)
        and np.isfinite(estimator_weight)
    ),
    "sample_weight and incorrect must be aligned, and estimator_weight must be finite",
)
@icontract.ensure(
    lambda result, sample_weight: _finite_nonnegative_vector(result, np.asarray(sample_weight, dtype=np.float64).shape),
    "updated classifier sample weights must remain finite, nonnegative, and shape-preserving",
)
def adaboost_classifier_sample_weight_update(
    sample_weight: NDArray[np.float64],
    incorrect: NDArray[np.bool_] | NDArray[np.int64],
    estimator_weight: float,
) -> NDArray[np.float64]:
    """Update AdaBoostClassifier sample weights for one nonterminal boosting stage."""
    weight_values = np.asarray(sample_weight, dtype=np.float64)
    incorrect_values = np.asarray(incorrect, dtype=np.float64)
    with np.errstate(divide="ignore"):
        updated = np.exp(np.log(weight_values) + estimator_weight * incorrect_values * (weight_values > 0.0))
    return np.asarray(updated, dtype=np.float64)


@register_atom(witness_adaboost_regressor_loss_vector)
@icontract.require(
    lambda absolute_errors, sample_weight, loss: (
        _weight_vector(sample_weight)
        and _positive_int(int(np.asarray(sample_weight, dtype=np.float64).shape[0]))
        and np.asarray(absolute_errors, dtype=np.float64).ndim == 1
        and np.asarray(absolute_errors, dtype=np.float64).shape == np.asarray(sample_weight, dtype=np.float64).shape
        and np.all(np.isfinite(np.asarray(absolute_errors, dtype=np.float64)))
        and np.all(np.asarray(absolute_errors, dtype=np.float64) >= 0.0)
        and loss in {"linear", "square", "exponential"}
    ),
    "absolute_errors must be a finite nonnegative vector aligned with sample_weight, and loss must be linear, square, or exponential",
)
@icontract.ensure(
    lambda result, absolute_errors: _bounded_loss_vector(result, absolute_errors),
    "regressor loss vector must preserve shape and stay within [0, 1]",
)
def adaboost_regressor_loss_vector(
    absolute_errors: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    loss: AdaBoostR2Loss,
) -> NDArray[np.float64]:
    """Build AdaBoost.R2's normalized per-sample loss vector from absolute errors."""
    error_values = np.asarray(absolute_errors, dtype=np.float64)
    weight_values = np.asarray(sample_weight, dtype=np.float64)
    mask = weight_values > 0.0
    result = np.zeros_like(error_values, dtype=np.float64)
    masked_errors = error_values[mask]
    error_max = float(masked_errors.max())
    if error_max != 0.0:
        result[mask] = masked_errors / error_max
    if loss == "square":
        result[mask] **= 2
    elif loss == "exponential":
        result[mask] = 1.0 - np.exp(-result[mask])
    return np.asarray(result, dtype=np.float64)


@register_atom(witness_adaboost_regressor_estimator_error)
@icontract.require(
    lambda loss_vector, sample_weight: _matching_regressor_inputs(loss_vector, sample_weight),
    "loss_vector must be aligned with a normalized finite nonnegative sample-weight vector",
)
@icontract.ensure(
    lambda result: np.isfinite(result) and 0.0 <= float(result) <= 1.0,
    "regressor stage error must be finite and lie in [0, 1]",
)
def adaboost_regressor_estimator_error(
    loss_vector: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
) -> float:
    """Compute AdaBoost.R2's weighted stage loss from the transformed loss vector."""
    return float(np.sum(np.asarray(sample_weight, dtype=np.float64) * np.asarray(loss_vector, dtype=np.float64)))


@register_atom(witness_adaboost_regressor_beta)
@icontract.require(
    lambda estimator_error: np.isfinite(estimator_error) and 0.0 < estimator_error < 0.5,
    "estimator_error must lie strictly between 0 and 0.5",
)
@icontract.ensure(lambda result: np.isfinite(result) and 0.0 < float(result) < 1.0, "beta must be finite and lie in (0, 1)")
def adaboost_regressor_beta(estimator_error: float) -> float:
    """Compute AdaBoost.R2's beta value from a valid stage error."""
    return float(estimator_error / (1.0 - estimator_error))


@register_atom(witness_adaboost_regressor_estimator_weight)
@icontract.require(
    lambda beta, learning_rate: np.isfinite(beta) and 0.0 < beta < 1.0 and np.isfinite(learning_rate) and learning_rate > 0.0,
    "beta must lie in (0, 1) and learning_rate must be positive",
)
@icontract.ensure(lambda result: np.isfinite(result), "regressor estimator weight must be finite")
def adaboost_regressor_estimator_weight(
    beta: float,
    learning_rate: float,
) -> float:
    """Compute AdaBoost.R2's estimator weight from beta."""
    return float(learning_rate * np.log(1.0 / beta))


@register_atom(witness_adaboost_regressor_sample_weight_update)
@icontract.require(
    lambda sample_weight, loss_vector, beta, learning_rate: (
        _matching_regressor_inputs(loss_vector, sample_weight)
        and np.isfinite(beta)
        and 0.0 < beta < 1.0
        and np.isfinite(learning_rate)
        and learning_rate > 0.0
    ),
    "sample_weight and loss_vector must be aligned, beta must lie in (0, 1), and learning_rate must be positive",
)
@icontract.ensure(
    lambda result, sample_weight: _finite_nonnegative_vector(result, np.asarray(sample_weight, dtype=np.float64).shape),
    "updated regressor sample weights must remain finite, nonnegative, and shape-preserving",
)
def adaboost_regressor_sample_weight_update(
    sample_weight: NDArray[np.float64],
    loss_vector: NDArray[np.float64],
    beta: float,
    learning_rate: float,
) -> NDArray[np.float64]:
    """Update AdaBoost.R2 sample weights for one nonterminal boosting stage."""
    weight_values = np.asarray(sample_weight, dtype=np.float64).copy()
    loss_values = np.asarray(loss_vector, dtype=np.float64)
    mask = weight_values > 0.0
    weight_values[mask] *= np.power(beta, (1.0 - loss_values[mask]) * learning_rate)
    return np.asarray(weight_values, dtype=np.float64)
