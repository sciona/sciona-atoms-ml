"""Ghost witnesses for sklearn forest OOB prediction helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_forest_classifier_oob_prediction_block(
    prediction: AbstractArray,
) -> AbstractArray:
    """Describe the classifier OOB prediction tensor for one forest tree."""
    if len(prediction.shape) not in {2, 3}:
        raise ValueError("prediction must be a matrix or rank-3 tensor")
    if len(prediction.shape) == 2:
        return AbstractArray(
            shape=(prediction.shape[0], prediction.shape[1], 1),
            dtype="float64",
            min_val=0.0,
        )
    return AbstractArray(
        shape=(prediction.shape[1], prediction.shape[2], prediction.shape[0]),
        dtype="float64",
        min_val=0.0,
    )


def witness_forest_regressor_oob_prediction_block(
    prediction: AbstractArray,
) -> AbstractArray:
    """Describe the regressor OOB prediction tensor for one forest tree."""
    if len(prediction.shape) not in {1, 2}:
        raise ValueError("prediction must be a vector or matrix")
    if len(prediction.shape) == 1:
        return AbstractArray(shape=(prediction.shape[0], 1, 1), dtype="float64")
    return AbstractArray(shape=(prediction.shape[0], 1, prediction.shape[1]), dtype="float64")


def witness_forest_oob_prediction_totals(
    prediction_blocks: tuple[AbstractArray, ...],
    unsampled_index_blocks: tuple[AbstractArray, ...],
    n_samples: int,
    prediction_width: int,
    n_outputs: int,
) -> AbstractArray:
    """Describe the summed held-out prediction tensor before dividing by the counts."""
    if n_samples < 1 or prediction_width < 1 or n_outputs < 1:
        raise ValueError("n_samples, prediction_width, and n_outputs must be positive")
    return AbstractArray(shape=(n_samples, prediction_width, n_outputs), dtype="float64")


def witness_forest_oob_prediction_counts(
    unsampled_index_blocks: tuple[AbstractArray, ...],
    n_samples: int,
    n_outputs: int,
) -> AbstractArray:
    """Describe per-sample forest OOB prediction counts before zero-count patching."""
    if n_samples < 1 or n_outputs < 1:
        raise ValueError("n_samples and n_outputs must be positive")
    return AbstractArray(shape=(n_samples, n_outputs), dtype="int64", min_val=0.0)


def witness_forest_oob_uncovered_mask(
    prediction_counts: AbstractArray,
) -> AbstractArray:
    """Describe which samples have no forest OOB predictions."""
    if len(prediction_counts.shape) != 2:
        raise ValueError("prediction_counts must be a matrix")
    return AbstractArray(shape=(prediction_counts.shape[0],), dtype="bool")


def witness_forest_oob_average_predictions(
    prediction_totals: AbstractArray,
    prediction_counts: AbstractArray,
) -> AbstractArray:
    """Describe averaged forest OOB predictions after zero-count patching."""
    if len(prediction_totals.shape) != 3 or len(prediction_counts.shape) != 2:
        raise ValueError("prediction_totals must be rank-3 and prediction_counts rank-2")
    return AbstractArray(shape=prediction_totals.shape, dtype="float64")
