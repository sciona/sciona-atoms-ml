"""Estimator-independent AdaBoost aggregation atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_adaboost_classifier_decision_function,
    witness_adaboost_classifier_probabilities_from_decision,
    witness_adaboost_regressor_weighted_median,
)


def _classes_valid(classes: NDArray[np.float64]) -> bool:
    values = np.asarray(classes, dtype=np.float64)
    return bool(
        values.ndim == 1
        and values.shape[0] >= 1
        and np.all(np.isfinite(values))
        and np.unique(values).shape[0] == values.shape[0]
    )


def _nonempty_weight_vector(estimator_weights: tuple[float, ...], n_estimators: int) -> bool:
    values = np.asarray(estimator_weights, dtype=np.float64)
    return bool(
        values.ndim == 1
        and values.shape[0] == n_estimators
        and n_estimators >= 1
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.sum(values) > 0.0
    )


def _encoded_predictions_valid(
    encoded_predictions: NDArray[np.int64],
    classes: NDArray[np.float64],
    estimator_weights: tuple[float, ...],
) -> bool:
    predictions = np.asarray(encoded_predictions)
    class_values = np.asarray(classes)
    return bool(
        predictions.ndim == 2
        and predictions.shape[0] >= 1
        and predictions.shape[1] >= 1
        and np.issubdtype(predictions.dtype, np.integer)
        and _classes_valid(classes)
        and _nonempty_weight_vector(estimator_weights, int(predictions.shape[1]))
        and np.all(predictions >= 0)
        and np.all(predictions < class_values.shape[0])
    )


def _decision_shape_valid(decision: NDArray[np.float64], n_classes: int) -> bool:
    values = np.asarray(decision, dtype=np.float64)
    if not isinstance(n_classes, int) or isinstance(n_classes, bool) or n_classes < 1:
        return False
    if n_classes == 1:
        return bool(values.ndim == 2 and values.shape[1] == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)))
    if n_classes == 2:
        return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)))
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] == n_classes and np.all(np.isfinite(values)))


def _decision_result_valid(
    result: NDArray[np.float64],
    encoded_predictions: NDArray[np.int64],
    classes: NDArray[np.float64],
) -> bool:
    predictions = np.asarray(encoded_predictions)
    n_samples = int(predictions.shape[0])
    n_classes = int(np.asarray(classes).shape[0])
    values = np.asarray(result, dtype=np.float64)
    if n_classes == 1:
        return bool(values.shape == (n_samples, 1) and np.all(values == 0.0))
    if n_classes == 2:
        return bool(values.shape == (n_samples,) and np.all(np.isfinite(values)))
    return bool(values.shape == (n_samples, n_classes) and np.all(np.isfinite(values)))


def _probability_result_valid(result: NDArray[np.float64], decision: NDArray[np.float64], n_classes: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    decision_values = np.asarray(decision, dtype=np.float64)
    n_samples = int(decision_values.shape[0])
    return bool(
        values.shape == (n_samples, n_classes)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _regressor_predictions_valid(
    predictions: NDArray[np.float64],
    estimator_weights: tuple[float, ...],
) -> bool:
    values = np.asarray(predictions, dtype=np.float64)
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.all(np.isfinite(values))
        and _nonempty_weight_vector(estimator_weights, int(values.shape[1]))
    )


def _median_result_valid(result: NDArray[np.float64], predictions: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    prediction_values = np.asarray(predictions, dtype=np.float64)
    return bool(values.shape == (prediction_values.shape[0],) and np.all(np.isfinite(values)))


def _softmax_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return np.asarray(exp_values / np.sum(exp_values, axis=1, keepdims=True), dtype=np.float64)


@register_atom(witness_adaboost_classifier_decision_function)
@icontract.require(
    lambda encoded_predictions, classes, estimator_weights: _encoded_predictions_valid(
        encoded_predictions,
        classes,
        estimator_weights,
    ),
    "encoded_predictions must be a sample-by-estimator integer matrix indexing the classes vector, and estimator_weights must be a finite nonnegative vector with positive total weight",
)
@icontract.ensure(
    lambda result, encoded_predictions, classes: _decision_result_valid(result, encoded_predictions, classes),
    "decision output must match sklearn's binary or multiclass AdaBoost shape",
)
def adaboost_classifier_decision_function(
    encoded_predictions: NDArray[np.int64],
    classes: NDArray[np.float64],
    estimator_weights: tuple[float, ...],
) -> NDArray[np.float64]:
    """Aggregate encoded classifier predictions into sklearn's AdaBoost decision output."""
    predictions = np.asarray(encoded_predictions, dtype=np.int64)
    class_values = np.asarray(classes, dtype=np.float64)
    weight_values = np.asarray(estimator_weights, dtype=np.float64)
    n_samples, n_estimators = predictions.shape
    n_classes = int(class_values.shape[0])

    if n_classes == 1:
        return np.zeros((n_samples, 1), dtype=np.float64)

    scores = np.zeros((n_samples, n_classes), dtype=np.float64)
    for column in range(n_estimators):
        weight = float(weight_values[column])
        scores += -weight / float(n_classes - 1)
        scores[np.arange(n_samples), predictions[:, column]] += weight * float(n_classes) / float(n_classes - 1)

    scores /= np.sum(weight_values)
    if n_classes == 2:
        scores[:, 0] *= -1.0
        return np.asarray(np.sum(scores, axis=1), dtype=np.float64)
    return np.asarray(scores, dtype=np.float64)


@register_atom(witness_adaboost_classifier_probabilities_from_decision)
@icontract.require(
    lambda decision, n_classes: _decision_shape_valid(decision, n_classes),
    "decision must match sklearn AdaBoost's binary or multiclass decision shape",
)
@icontract.ensure(
    lambda result, decision, n_classes: _probability_result_valid(result, decision, n_classes),
    "probabilities must be finite, nonnegative, and normalized per sample",
)
def adaboost_classifier_probabilities_from_decision(
    decision: NDArray[np.float64],
    n_classes: int,
) -> NDArray[np.float64]:
    """Convert AdaBoost decision outputs to class probabilities the way sklearn does."""
    if n_classes == 1:
        decision_values = np.asarray(decision, dtype=np.float64)
        return np.ones((decision_values.shape[0], 1), dtype=np.float64)

    if n_classes == 2:
        decision_matrix = np.vstack([-np.asarray(decision, dtype=np.float64), np.asarray(decision, dtype=np.float64)]).T / 2.0
    else:
        decision_matrix = np.asarray(decision, dtype=np.float64) / float(n_classes - 1)
    return _softmax_rows(np.asarray(decision_matrix, dtype=np.float64))


@register_atom(witness_adaboost_regressor_weighted_median)
@icontract.require(
    lambda predictions, estimator_weights: _regressor_predictions_valid(predictions, estimator_weights),
    "predictions must be a finite sample-by-estimator matrix and estimator_weights must be a matching finite nonnegative vector with positive total weight",
)
@icontract.ensure(
    lambda result, predictions: _median_result_valid(result, predictions),
    "weighted-median aggregation must return one finite regression value per sample",
)
def adaboost_regressor_weighted_median(
    predictions: NDArray[np.float64],
    estimator_weights: tuple[float, ...],
) -> NDArray[np.float64]:
    """Select AdaBoost.R2's weighted-median regression prediction for each sample."""
    prediction_values = np.asarray(predictions, dtype=np.float64)
    weight_values = np.asarray(estimator_weights, dtype=np.float64)
    sorted_idx = np.argsort(prediction_values, axis=1)
    weight_cdf = np.cumsum(weight_values[sorted_idx], axis=1)
    median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
    median_idx = median_or_above.argmax(axis=1)
    median_estimators = sorted_idx[np.arange(prediction_values.shape[0]), median_idx]
    return np.asarray(prediction_values[np.arange(prediction_values.shape[0]), median_estimators], dtype=np.float64)
