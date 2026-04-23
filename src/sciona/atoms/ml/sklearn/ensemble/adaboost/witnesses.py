"""Ghost witnesses for sklearn AdaBoost aggregation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_adaboost_classifier_decision_function(
    encoded_predictions: AbstractArray,
    classes: AbstractArray,
    estimator_weights: tuple[float, ...],
) -> AbstractArray:
    """Describe sklearn AdaBoost classifier decision outputs from encoded predictions."""
    if len(encoded_predictions.shape) != 2:
        raise ValueError("encoded_predictions must be a 2D matrix")
    if len(classes.shape) != 1:
        raise ValueError("classes must be a 1D vector")
    n_samples = encoded_predictions.shape[0]
    n_classes = int(classes.shape[0])
    if n_classes <= 0 or len(estimator_weights) < 1:
        raise ValueError("classes and estimator_weights must be nonempty")
    if n_classes == 2:
        return AbstractArray(shape=(n_samples,), dtype="float64")
    return AbstractArray(shape=(n_samples, n_classes), dtype="float64")


def witness_adaboost_classifier_probabilities_from_decision(
    decision: AbstractArray,
    n_classes: int,
) -> AbstractArray:
    """Describe AdaBoost class-probability outputs from decision values."""
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    if n_classes == 2:
        if len(decision.shape) != 1:
            raise ValueError("binary decision must be one-dimensional")
        return AbstractArray(shape=(decision.shape[0], 2), dtype="float64", min_val=0.0, max_val=1.0)
    if len(decision.shape) != 2:
        raise ValueError("multiclass decision must be two-dimensional")
    return AbstractArray(shape=(decision.shape[0], n_classes), dtype="float64", min_val=0.0, max_val=1.0)


def witness_adaboost_regressor_weighted_median(
    predictions: AbstractArray,
    estimator_weights: tuple[float, ...],
) -> AbstractArray:
    """Describe one weighted-median regression value per sample."""
    if len(predictions.shape) != 2:
        raise ValueError("predictions must be a 2D matrix")
    if len(estimator_weights) < 1:
        raise ValueError("estimator_weights must be nonempty")
    return AbstractArray(shape=(predictions.shape[0],), dtype="float64")
