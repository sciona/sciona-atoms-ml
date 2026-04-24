"""Ghost witnesses for sklearn bagging OOB scoring helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_bagging_oob_uncovered_mask(
    sample_index_blocks: tuple[AbstractArray, ...],
    n_samples: int,
) -> AbstractArray:
    """Describe the mask of samples never left out by any bagging estimator."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    for block in sample_index_blocks:
        if len(block.shape) != 1 or int(block.shape[0]) < 1:
            raise ValueError("sample index blocks must be nonempty 1D arrays")
    return AbstractArray(shape=(n_samples,), dtype="bool")


def witness_bagging_classifier_oob_accuracy(
    y_encoded: AbstractArray,
    prediction_totals: AbstractArray,
) -> float:
    """Describe the fraction of rows whose largest class total matches the label."""
    if len(y_encoded.shape) != 1 or int(y_encoded.shape[0]) < 1:
        raise ValueError("y_encoded must be a nonempty vector")
    if len(prediction_totals.shape) != 2 or int(prediction_totals.shape[0]) != int(y_encoded.shape[0]):
        raise ValueError("prediction_totals must be a sample-aligned 2D matrix")
    return 0.0


def witness_bagging_regressor_oob_r2(
    y_true: AbstractArray,
    predictions: AbstractArray,
) -> float:
    """Describe how much of the target variation the predictions explain."""
    if len(y_true.shape) != 1 or len(predictions.shape) != 1:
        raise ValueError("y_true and predictions must be 1D")
    if int(y_true.shape[0]) < 1 or int(y_true.shape[0]) != int(predictions.shape[0]):
        raise ValueError("y_true and predictions must be nonempty and aligned")
    return 0.0
