"""Functions for turning averaged forest outputs into public results."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, r2_score

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_forest_classifier_oob_accuracy,
    witness_forest_classifier_oob_decision_function,
    witness_forest_regressor_oob_prediction,
    witness_forest_regressor_oob_r2,
)


def _averaged_classifier_predictions_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    row_sums = np.sum(array, axis=1)
    return bool(
        array.ndim == 3
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and array.shape[2] >= 1
        and np.all(np.isfinite(array))
        and np.all(array >= 0.0)
        and np.all(np.isclose(row_sums, 1.0) | np.isclose(row_sums, 0.0))
    )


def _classifier_decision_function_valid(
    decision_function: object,
    y_true: object,
) -> bool:
    try:
        decision = np.asarray(decision_function, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    targets = np.asarray(y_true)
    if not (
        targets.ndim in {1, 2}
        and decision.ndim in {2, 3}
        and targets.shape[0] >= 1
        and targets.shape[0] == decision.shape[0]
        and np.issubdtype(targets.dtype, np.integer)
        and np.all(targets >= 0)
        and np.all(np.isfinite(decision))
        and np.all(decision >= 0.0)
    ):
        return False
    if decision.ndim == 2:
        row_sums = np.sum(decision, axis=1)
        return bool(
            targets.ndim == 1
            and decision.shape[1] >= 1
            and np.all(np.isclose(row_sums, 1.0) | np.isclose(row_sums, 0.0))
            and np.all(targets < decision.shape[1])
        )
    row_sums = np.sum(decision, axis=1)
    return bool(
        targets.ndim == 2
        and decision.shape[1] >= 1
        and decision.shape[2] == targets.shape[1]
        and np.all(np.isclose(row_sums, 1.0) | np.isclose(row_sums, 0.0))
        and np.all(targets < decision.shape[1])
    )


def _classifier_decision_result_valid(result: object, averaged_predictions: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    averaged = np.asarray(averaged_predictions, dtype=np.float64)
    if averaged.shape[2] == 1:
        return bool(values.shape == (averaged.shape[0], averaged.shape[1]) and np.all(np.isfinite(values)))
    return bool(values.shape == averaged.shape and np.all(np.isfinite(values)))


def _averaged_regressor_predictions_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 3
        and array.shape[0] >= 1
        and array.shape[1] == 1
        and array.shape[2] >= 1
        and np.all(np.isfinite(array))
    )


def _regressor_prediction_result_valid(result: object, averaged_predictions: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    averaged = np.asarray(averaged_predictions, dtype=np.float64)
    if averaged.shape[2] == 1:
        return bool(values.shape == (averaged.shape[0],) and np.all(np.isfinite(values)))
    return bool(values.shape == (averaged.shape[0], averaged.shape[2]) and np.all(np.isfinite(values)))


def _aligned_regression_arrays(y_true: object, prediction: object) -> bool:
    try:
        truth = np.asarray(y_true, dtype=np.float64)
        preds = np.asarray(prediction, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        truth.ndim in {1, 2}
        and preds.ndim in {1, 2}
        and truth.shape == preds.shape
        and truth.shape[0] >= 1
        and np.all(np.isfinite(truth))
        and np.all(np.isfinite(preds))
    )


@register_atom(witness_forest_classifier_oob_decision_function)
@icontract.require(
    lambda averaged_predictions: _averaged_classifier_predictions_valid(averaged_predictions),
    "averaged_predictions must be a finite normalized sample-by-class-by-output tensor",
)
@icontract.ensure(
    lambda result, averaged_predictions: _classifier_decision_result_valid(result, averaged_predictions),
    "classifier OOB decision function must match sklearn's squeezed public shape",
)
def forest_classifier_oob_decision_function(
    averaged_predictions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert the averaged classifier OOB tensor into sklearn's public decision-function shape."""
    decision = np.asarray(averaged_predictions, dtype=np.float64)
    if decision.shape[-1] == 1:
        decision = decision.squeeze(axis=-1)
    return np.asarray(decision, dtype=np.float64)


@register_atom(witness_forest_classifier_oob_accuracy)
@icontract.require(
    lambda y_true, decision_function: _classifier_decision_function_valid(decision_function, y_true),
    "y_true and decision_function must align as encoded classifier targets and normalized probabilities",
)
@icontract.ensure(
    lambda result: isinstance(result, float) and np.isfinite(result) and 0.0 <= result <= 1.0,
    "classifier OOB accuracy must be a finite value in [0, 1]",
)
def forest_classifier_oob_accuracy(
    y_true: NDArray[np.int64],
    decision_function: NDArray[np.float64],
) -> float:
    """Score classifier OOB predictions the way sklearn does after public-shape conversion."""
    targets = np.asarray(y_true, dtype=np.int64)
    decision = np.asarray(decision_function, dtype=np.float64)
    return float(accuracy_score(targets, np.argmax(decision, axis=1)))


@register_atom(witness_forest_regressor_oob_prediction)
@icontract.require(
    lambda averaged_predictions: _averaged_regressor_predictions_valid(averaged_predictions),
    "averaged_predictions must be a finite sample-by-1-by-output tensor",
)
@icontract.ensure(
    lambda result, averaged_predictions: _regressor_prediction_result_valid(result, averaged_predictions),
    "regressor OOB prediction must match sklearn's squeezed public shape",
)
def forest_regressor_oob_prediction(
    averaged_predictions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert the averaged regressor OOB tensor into sklearn's public prediction shape."""
    prediction = np.asarray(averaged_predictions, dtype=np.float64).squeeze(axis=1)
    if prediction.shape[-1] == 1:
        prediction = prediction.squeeze(axis=-1)
    return np.asarray(prediction, dtype=np.float64)


@register_atom(witness_forest_regressor_oob_r2)
@icontract.require(
    lambda y_true, prediction: _aligned_regression_arrays(y_true, prediction),
    "y_true and prediction must be aligned finite regression arrays",
)
@icontract.ensure(
    lambda result: isinstance(result, float) and np.isfinite(result),
    "regressor OOB r2 must be a finite value",
)
def forest_regressor_oob_r2(
    y_true: NDArray[np.float64],
    prediction: NDArray[np.float64],
) -> float:
    """Score regressor OOB predictions the way sklearn does after public-shape conversion."""
    return float(r2_score(np.asarray(y_true, dtype=np.float64), np.asarray(prediction, dtype=np.float64)))
