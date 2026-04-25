"""Ghost witnesses for sklearn forest OOB postprocessing helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_forest_classifier_oob_decision_function(
    averaged_predictions: AbstractArray,
) -> AbstractArray:
    """Describe the public classifier OOB decision-function shape after squeezing."""
    if len(averaged_predictions.shape) != 3:
        raise ValueError("averaged_predictions must be rank-3")
    if int(averaged_predictions.shape[1]) < 1 or int(averaged_predictions.shape[2]) < 1:
        raise ValueError("class and output axes must be positive")
    if int(averaged_predictions.shape[2]) == 1:
        return AbstractArray(
            shape=(averaged_predictions.shape[0], averaged_predictions.shape[1]),
            dtype="float64",
            min_val=0.0,
        )
    return AbstractArray(shape=averaged_predictions.shape, dtype="float64", min_val=0.0)


def witness_forest_classifier_oob_accuracy(
    y_true: AbstractArray,
    decision_function: AbstractArray,
) -> float:
    """Describe classifier OOB accuracy from encoded targets and the public decision function."""
    if len(y_true.shape) not in {1, 2}:
        raise ValueError("y_true must be a vector or matrix")
    if len(decision_function.shape) not in {2, 3}:
        raise ValueError("decision_function must be a matrix or rank-3 tensor")
    if int(y_true.shape[0]) != int(decision_function.shape[0]):
        raise ValueError("y_true and decision_function must agree on samples")
    return 0.0


def witness_forest_regressor_oob_prediction(
    averaged_predictions: AbstractArray,
) -> AbstractArray:
    """Describe the public regressor OOB prediction shape after squeezing."""
    if len(averaged_predictions.shape) != 3:
        raise ValueError("averaged_predictions must be rank-3")
    if int(averaged_predictions.shape[1]) != 1 or int(averaged_predictions.shape[2]) < 1:
        raise ValueError("averaged_predictions must have width 1 and a positive output axis")
    if int(averaged_predictions.shape[2]) == 1:
        return AbstractArray(shape=(averaged_predictions.shape[0],), dtype="float64")
    return AbstractArray(shape=(averaged_predictions.shape[0], averaged_predictions.shape[2]), dtype="float64")


def witness_forest_regressor_oob_r2(
    y_true: AbstractArray,
    prediction: AbstractArray,
) -> float:
    """Describe regressor OOB r2 from aligned public target and prediction arrays."""
    if len(y_true.shape) not in {1, 2} or len(prediction.shape) not in {1, 2}:
        raise ValueError("y_true and prediction must be vectors or matrices")
    if tuple(y_true.shape) != tuple(prediction.shape):
        raise ValueError("y_true and prediction must have identical shapes")
    return 0.0
