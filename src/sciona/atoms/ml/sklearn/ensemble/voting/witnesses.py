"""Ghost witnesses for sklearn voting aggregation helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_weights(weights: tuple[float, ...] | None, n_estimators: int) -> None:
    if weights is not None and len(weights) != n_estimators:
        raise ValueError("weights length must match estimator count")


def witness_voting_classifier_hard_labels(
    encoded_predictions: AbstractArray,
    classes: AbstractArray,
    *,
    weights: tuple[float, ...] | None = None,
) -> AbstractArray:
    """Describe one hard-vote class label per sample."""
    if len(encoded_predictions.shape) != 2:
        raise ValueError("encoded_predictions must be 2D")
    if len(classes.shape) != 1:
        raise ValueError("classes must be 1D")
    n_samples, n_estimators = int(encoded_predictions.shape[0]), int(encoded_predictions.shape[1])
    if n_samples < 1 or n_estimators < 1 or int(classes.shape[0]) < 1:
        raise ValueError("voting inputs must be nonempty")
    _check_weights(weights, n_estimators)
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_voting_classifier_soft_probabilities(
    probabilities: AbstractArray,
    *,
    weights: tuple[float, ...] | None = None,
) -> AbstractArray:
    """Describe averaged class probabilities for each sample."""
    if len(probabilities.shape) != 3:
        raise ValueError("probabilities must be 3D")
    n_estimators = int(probabilities.shape[0])
    n_samples = int(probabilities.shape[1])
    n_classes = int(probabilities.shape[2])
    if n_estimators < 1 or n_samples < 1 or n_classes < 1:
        raise ValueError("probability tensor must be nonempty")
    _check_weights(weights, n_estimators)
    return AbstractArray(shape=(n_samples, n_classes), dtype="float64", min_val=0.0, max_val=1.0)


def witness_voting_regressor_average(
    predictions: AbstractArray,
    *,
    weights: tuple[float, ...] | None = None,
) -> AbstractArray:
    """Describe averaged regressor predictions for each sample."""
    if len(predictions.shape) != 2:
        raise ValueError("predictions must be 2D")
    n_samples, n_estimators = int(predictions.shape[0]), int(predictions.shape[1])
    if n_samples < 1 or n_estimators < 1:
        raise ValueError("prediction matrix must be nonempty")
    _check_weights(weights, n_estimators)
    return AbstractArray(shape=(n_samples,), dtype="float64")
