"""Ghost witnesses for sklearn forest aggregation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_forest_classifier_average_probabilities(
    probabilities: AbstractArray,
) -> AbstractArray:
    """Describe mean class probabilities over a tree axis."""
    if len(probabilities.shape) != 3:
        raise ValueError("probabilities must be a 3D tensor")
    return AbstractArray(shape=(probabilities.shape[1], probabilities.shape[2]), dtype="float64", min_val=0.0, max_val=1.0)


def witness_forest_classifier_labels_from_probabilities(
    probabilities: AbstractArray,
    classes: AbstractArray,
) -> AbstractArray:
    """Describe one predicted class label per sample."""
    if len(probabilities.shape) != 2:
        raise ValueError("probabilities must be a 2D matrix")
    if len(classes.shape) != 1:
        raise ValueError("classes must be a 1D vector")
    return AbstractArray(shape=(probabilities.shape[0],), dtype="float64")


def witness_forest_regressor_average_predictions(
    predictions: AbstractArray,
) -> AbstractArray:
    """Describe mean regression predictions over a tree axis."""
    if len(predictions.shape) == 2:
        return AbstractArray(shape=(predictions.shape[1],), dtype="float64")
    if len(predictions.shape) == 3:
        return AbstractArray(shape=(predictions.shape[1], predictions.shape[2]), dtype="float64")
    raise ValueError("predictions must be 2D or 3D")
