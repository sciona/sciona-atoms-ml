"""Estimator-independent voting aggregation atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_voting_classifier_hard_labels,
    witness_voting_classifier_soft_probabilities,
    witness_voting_regressor_average,
)


def _classes_valid(classes: NDArray[np.float64]) -> bool:
    values = np.asarray(classes, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)) and np.unique(values).shape[0] == values.shape[0])


def _encoded_predictions_valid(encoded_predictions: NDArray[np.int64], classes: NDArray[np.float64]) -> bool:
    predictions = np.asarray(encoded_predictions)
    class_values = np.asarray(classes)
    return bool(
        predictions.ndim == 2
        and predictions.shape[0] >= 1
        and predictions.shape[1] >= 1
        and np.issubdtype(predictions.dtype, np.integer)
        and _classes_valid(classes)
        and np.all(predictions >= 0)
        and np.all(predictions < class_values.shape[0])
    )


def _weights_valid(weights: tuple[float, ...] | None, n_estimators: int) -> bool:
    if weights is None:
        return True
    weight_values = np.asarray(weights, dtype=np.float64)
    return bool(
        weight_values.ndim == 1
        and weight_values.shape[0] == n_estimators
        and np.all(np.isfinite(weight_values))
        and np.all(weight_values >= 0.0)
        and np.sum(weight_values) > 0.0
    )


def _prediction_weights_valid(weights: tuple[float, ...] | None, encoded_predictions: NDArray[np.int64]) -> bool:
    values = np.asarray(encoded_predictions)
    return bool(values.ndim == 2 and _weights_valid(weights, int(values.shape[1])))


def _probability_tensor_valid(probabilities: NDArray[np.float64]) -> bool:
    values = np.asarray(probabilities, dtype=np.float64)
    return bool(
        values.ndim == 3
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and values.shape[2] >= 1
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=2), 1.0)
    )


def _probability_weights_valid(weights: tuple[float, ...] | None, probabilities: NDArray[np.float64]) -> bool:
    values = np.asarray(probabilities)
    return bool(values.ndim == 3 and _weights_valid(weights, int(values.shape[0])))


def _regressor_predictions_valid(predictions: NDArray[np.float64]) -> bool:
    values = np.asarray(predictions, dtype=np.float64)
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _regressor_weights_valid(weights: tuple[float, ...] | None, predictions: NDArray[np.float64]) -> bool:
    values = np.asarray(predictions)
    return bool(values.ndim == 2 and _weights_valid(weights, int(values.shape[1])))


def _hard_labels_valid(result: NDArray[np.float64], encoded_predictions: NDArray[np.int64], classes: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    predictions = np.asarray(encoded_predictions)
    return bool(values.shape == (predictions.shape[0],) and np.all(np.isin(values, class_values)))


def _soft_probabilities_valid(result: NDArray[np.float64], probabilities: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(probabilities, dtype=np.float64)
    return bool(
        values.shape == (input_values.shape[1], input_values.shape[2])
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _averages_valid(result: NDArray[np.float64], predictions: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(predictions, dtype=np.float64)
    return bool(values.shape == (input_values.shape[0],) and np.all(np.isfinite(values)))


def _weights_array(weights: tuple[float, ...] | None) -> NDArray[np.float64] | None:
    if weights is None:
        return None
    return np.asarray(weights, dtype=np.float64)


@register_atom(witness_voting_classifier_hard_labels)
@icontract.require(lambda encoded_predictions, classes: _encoded_predictions_valid(encoded_predictions, classes), "encoded predictions must index the class vector")
@icontract.require(lambda encoded_predictions, weights: _prediction_weights_valid(weights, encoded_predictions), "weights must match estimator count and have positive total weight")
@icontract.ensure(lambda result, encoded_predictions, classes: _hard_labels_valid(result, encoded_predictions, classes), "hard-vote labels must come from the class vector")
def voting_classifier_hard_labels(
    encoded_predictions: NDArray[np.int64],
    classes: NDArray[np.float64],
    *,
    weights: tuple[float, ...] | None = None,
) -> NDArray[np.float64]:
    """Choose each sample's class by sklearn's weighted hard-vote tie rule."""
    predictions = np.asarray(encoded_predictions, dtype=np.int64)
    class_values = np.asarray(classes, dtype=np.float64)
    weight_values = _weights_array(weights)

    encoded_labels = np.apply_along_axis(
        lambda row: np.argmax(np.bincount(row, weights=weight_values, minlength=class_values.shape[0])),
        axis=1,
        arr=predictions,
    )
    return np.asarray(class_values[encoded_labels], dtype=np.float64)


@register_atom(witness_voting_classifier_soft_probabilities)
@icontract.require(lambda probabilities: _probability_tensor_valid(probabilities), "probabilities must be per-estimator normalized class probabilities")
@icontract.require(lambda probabilities, weights: _probability_weights_valid(weights, probabilities), "weights must match estimator count and have positive total weight")
@icontract.ensure(lambda result, probabilities: _soft_probabilities_valid(result, probabilities), "averaged probabilities must stay normalized per sample")
def voting_classifier_soft_probabilities(
    probabilities: NDArray[np.float64],
    *,
    weights: tuple[float, ...] | None = None,
) -> NDArray[np.float64]:
    """Average per-estimator class probabilities the way sklearn soft voting does."""
    return np.asarray(
        np.average(np.asarray(probabilities, dtype=np.float64), axis=0, weights=_weights_array(weights)),
        dtype=np.float64,
    )


@register_atom(witness_voting_regressor_average)
@icontract.require(lambda predictions: _regressor_predictions_valid(predictions), "predictions must be a finite sample-by-estimator matrix")
@icontract.require(lambda predictions, weights: _regressor_weights_valid(weights, predictions), "weights must match estimator count and have positive total weight")
@icontract.ensure(lambda result, predictions: _averages_valid(result, predictions), "averaged predictions must have one value per sample")
def voting_regressor_average(
    predictions: NDArray[np.float64],
    *,
    weights: tuple[float, ...] | None = None,
) -> NDArray[np.float64]:
    """Average per-estimator regression predictions the way sklearn voting regression does."""
    return np.asarray(
        np.average(np.asarray(predictions, dtype=np.float64), axis=1, weights=_weights_array(weights)),
        dtype=np.float64,
    )
