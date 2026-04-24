"""Ghost witnesses for sklearn AdaBoost weight update helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_adaboost_classifier_estimator_error(
    incorrect: AbstractArray,
    sample_weight: AbstractArray,
) -> float:
    """Describe AdaBoost classifier stage error from a boolean mistake mask."""
    if len(incorrect.shape) != 1 or len(sample_weight.shape) != 1:
        raise ValueError("incorrect and sample_weight must be one-dimensional")
    if incorrect.shape[0] != sample_weight.shape[0]:
        raise ValueError("incorrect and sample_weight must have the same length")
    return 0.0


def witness_adaboost_classifier_estimator_weight(
    estimator_error: float,
    learning_rate: float,
    n_classes: int,
) -> float:
    """Describe AdaBoost classifier stage weight from the stage error."""
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    return 0.0


def witness_adaboost_classifier_sample_weight_update(
    sample_weight: AbstractArray,
    incorrect: AbstractArray,
    estimator_weight: float,
) -> AbstractArray:
    """Describe AdaBoost classifier sample-weight updates."""
    if len(sample_weight.shape) != 1 or len(incorrect.shape) != 1:
        raise ValueError("sample_weight and incorrect must be one-dimensional")
    if sample_weight.shape[0] != incorrect.shape[0]:
        raise ValueError("sample_weight and incorrect must have the same length")
    return AbstractArray(shape=sample_weight.shape, dtype="float64", min_val=0.0)


def witness_adaboost_regressor_loss_vector(
    absolute_errors: AbstractArray,
    sample_weight: AbstractArray,
    loss: str,
) -> AbstractArray:
    """Describe AdaBoost.R2's normalized per-sample loss vector."""
    if len(absolute_errors.shape) != 1 or len(sample_weight.shape) != 1:
        raise ValueError("absolute_errors and sample_weight must be one-dimensional")
    if absolute_errors.shape[0] != sample_weight.shape[0]:
        raise ValueError("absolute_errors and sample_weight must have the same length")
    return AbstractArray(shape=absolute_errors.shape, dtype="float64", min_val=0.0, max_val=1.0)


def witness_adaboost_regressor_estimator_error(
    loss_vector: AbstractArray,
    sample_weight: AbstractArray,
) -> float:
    """Describe AdaBoost.R2's weighted average stage loss."""
    if len(loss_vector.shape) != 1 or len(sample_weight.shape) != 1:
        raise ValueError("loss_vector and sample_weight must be one-dimensional")
    if loss_vector.shape[0] != sample_weight.shape[0]:
        raise ValueError("loss_vector and sample_weight must have the same length")
    return 0.0


def witness_adaboost_regressor_beta(estimator_error: float) -> float:
    """Describe the AdaBoost.R2 beta value from a valid stage error."""
    return 0.0


def witness_adaboost_regressor_estimator_weight(
    beta: float,
    learning_rate: float,
) -> float:
    """Describe the AdaBoost.R2 estimator weight from beta."""
    return 0.0


def witness_adaboost_regressor_sample_weight_update(
    sample_weight: AbstractArray,
    loss_vector: AbstractArray,
    beta: float,
    learning_rate: float,
) -> AbstractArray:
    """Describe AdaBoost.R2 sample-weight updates."""
    if len(sample_weight.shape) != 1 or len(loss_vector.shape) != 1:
        raise ValueError("sample_weight and loss_vector must be one-dimensional")
    if sample_weight.shape[0] != loss_vector.shape[0]:
        raise ValueError("sample_weight and loss_vector must have the same length")
    return AbstractArray(shape=sample_weight.shape, dtype="float64", min_val=0.0)
