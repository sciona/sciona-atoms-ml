"""Estimator-independent forest aggregation atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_forest_classifier_average_probabilities,
    witness_forest_classifier_labels_from_probabilities,
    witness_forest_regressor_average_predictions,
)


def _classes_valid(classes: NDArray[np.float64]) -> bool:
    values = np.asarray(classes, dtype=np.float64)
    return bool(
        values.ndim == 1
        and values.shape[0] >= 1
        and np.all(np.isfinite(values))
        and np.unique(values).shape[0] == values.shape[0]
    )


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


def _probabilities_match_classes(probabilities: NDArray[np.float64], classes: NDArray[np.float64]) -> bool:
    values = np.asarray(probabilities)
    class_values = np.asarray(classes)
    return bool(values.ndim == 3 and _classes_valid(classes) and values.shape[2] == class_values.shape[0])


def _average_probabilities_valid(result: NDArray[np.float64], probabilities: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(probabilities, dtype=np.float64)
    return bool(
        values.shape == (input_values.shape[1], input_values.shape[2])
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _aggregated_probabilities_valid(probabilities: NDArray[np.float64], classes: NDArray[np.float64]) -> bool:
    values = np.asarray(probabilities, dtype=np.float64)
    class_values = np.asarray(classes)
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
        and _classes_valid(classes)
        and values.shape[1] == class_values.shape[0]
    )


def _labels_valid(result: NDArray[np.float64], probabilities: NDArray[np.float64], classes: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    proba_values = np.asarray(probabilities)
    class_values = np.asarray(classes, dtype=np.float64)
    return bool(values.shape == (proba_values.shape[0],) and np.all(np.isin(values, class_values)))


def _regression_predictions_valid(predictions: NDArray[np.float64]) -> bool:
    values = np.asarray(predictions, dtype=np.float64)
    return bool(
        values.ndim in {2, 3}
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and (values.ndim == 2 or values.shape[2] >= 1)
        and np.all(np.isfinite(values))
    )


def _average_predictions_valid(result: NDArray[np.float64], predictions: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(predictions, dtype=np.float64)
    expected_shape = input_values.shape[1:] if input_values.ndim == 3 else (input_values.shape[1],)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


@register_atom(witness_forest_classifier_average_probabilities)
@icontract.require(
    lambda probabilities: _probability_tensor_valid(probabilities),
    "probabilities must be a tree-by-sample-by-class tensor of normalized class probabilities",
)
@icontract.ensure(
    lambda result, probabilities: _average_probabilities_valid(result, probabilities),
    "averaged probabilities must stay normalized per sample",
)
def forest_classifier_average_probabilities(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Average per-tree class probabilities the way sklearn forest classifiers do."""
    return np.asarray(np.mean(np.asarray(probabilities, dtype=np.float64), axis=0), dtype=np.float64)


@register_atom(witness_forest_classifier_labels_from_probabilities)
@icontract.require(
    lambda probabilities, classes: _aggregated_probabilities_valid(probabilities, classes),
    "probabilities must be a normalized sample-by-class matrix matching the classes vector",
)
@icontract.ensure(
    lambda result, probabilities, classes: _labels_valid(result, probabilities, classes),
    "predicted labels must come from the classes vector",
)
def forest_classifier_labels_from_probabilities(
    probabilities: NDArray[np.float64],
    classes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Choose each sample's class from aggregated forest probabilities."""
    values = np.asarray(probabilities, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    return np.asarray(class_values.take(np.argmax(values, axis=1), axis=0), dtype=np.float64)


@register_atom(witness_forest_regressor_average_predictions)
@icontract.require(
    lambda predictions: _regression_predictions_valid(predictions),
    "predictions must be a finite tree-by-sample matrix or tree-by-sample-by-output tensor",
)
@icontract.ensure(
    lambda result, predictions: _average_predictions_valid(result, predictions),
    "averaged predictions must preserve the sample and optional output axes",
)
def forest_regressor_average_predictions(
    predictions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Average per-tree regression predictions the way sklearn forest regressors do."""
    return np.asarray(np.mean(np.asarray(predictions, dtype=np.float64), axis=0), dtype=np.float64)
