"""Ghost witnesses for sklearn bagging out-of-bag aggregation helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_bagging_classifier_oob_probability_totals(
    probability_blocks: tuple[AbstractArray, ...],
    class_index_blocks: tuple[AbstractArray, ...],
    sample_index_blocks: tuple[AbstractArray, ...],
    n_samples: int,
    n_classes: int,
) -> AbstractArray:
    """Describe bagging classifier OOB class-total accumulation from probability blocks."""
    if n_samples < 1 or n_classes < 1:
        raise ValueError("n_samples and n_classes must be positive")
    return AbstractArray(shape=(n_samples, n_classes), dtype="float64", min_val=0.0)


def witness_bagging_classifier_oob_vote_totals(
    predicted_label_blocks: tuple[AbstractArray, ...],
    sample_index_blocks: tuple[AbstractArray, ...],
    n_samples: int,
    n_classes: int,
) -> AbstractArray:
    """Describe bagging classifier OOB vote accumulation from label blocks."""
    if n_samples < 1 or n_classes < 1:
        raise ValueError("n_samples and n_classes must be positive")
    return AbstractArray(shape=(n_samples, n_classes), dtype="float64", min_val=0.0)


def witness_bagging_classifier_oob_decision_function(
    prediction_totals: AbstractArray,
) -> AbstractArray:
    """Describe per-sample class shares after normalizing held-out totals."""
    if len(prediction_totals.shape) != 2:
        raise ValueError("prediction_totals must be a matrix")
    return AbstractArray(shape=prediction_totals.shape, dtype="float64")


def witness_bagging_classifier_oob_label_indices(
    prediction_totals: AbstractArray,
) -> AbstractArray:
    """Describe OOB argmax label indices from classifier totals."""
    if len(prediction_totals.shape) != 2:
        raise ValueError("prediction_totals must be a matrix")
    return AbstractArray(shape=(prediction_totals.shape[0],), dtype="int64", min_val=0.0)


def witness_bagging_regressor_oob_predictions(
    prediction_blocks: tuple[AbstractArray, ...],
    sample_index_blocks: tuple[AbstractArray, ...],
    n_samples: int,
) -> AbstractArray:
    """Describe one held-out average prediction for each sample."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return AbstractArray(shape=(n_samples,), dtype="float64")
