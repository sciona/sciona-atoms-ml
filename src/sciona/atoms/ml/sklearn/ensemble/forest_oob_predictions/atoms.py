"""Forest OOB prediction helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_forest_classifier_oob_prediction_block,
    witness_forest_oob_average_predictions,
    witness_forest_oob_prediction_counts,
    witness_forest_oob_prediction_totals,
    witness_forest_oob_uncovered_mask,
    witness_forest_regressor_oob_prediction_block,
)

PredictionBlock = NDArray[np.float64]
PredictionBlockTuple = tuple[PredictionBlock, ...]
IndexTuple = tuple[NDArray[np.int64], ...]


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _classifier_prediction_valid(prediction: object) -> bool:
    try:
        values = np.asarray(prediction, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim in {2, 3}
        and values.shape[0] >= 1
        and values.shape[1] >= 0
        and (values.ndim == 2 or values.shape[2] >= 1)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and (
            (values.ndim == 2 and np.allclose(np.sum(values, axis=1), 1.0))
            or (
                values.ndim == 3
                and np.allclose(np.sum(values, axis=2), 1.0)
            )
        )
    )


def _regressor_prediction_valid(prediction: object) -> bool:
    try:
        values = np.asarray(prediction, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim in {1, 2}
        and values.shape[0] >= 0
        and (values.ndim == 1 or values.shape[1] >= 1)
        and np.all(np.isfinite(values))
    )


def _prediction_block_valid(
    block: object,
    *,
    expected_width: int,
    n_outputs: int,
) -> bool:
    try:
        values = np.asarray(block, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        _positive_int(expected_width)
        and _positive_int(n_outputs)
        and values.ndim == 3
        and values.shape[0] >= 0
        and values.shape[1] == expected_width
        and values.shape[2] == n_outputs
        and np.all(np.isfinite(values))
    )


def _unsampled_index_block_valid(block: object, n_samples: int) -> bool:
    values = np.asarray(block)
    return bool(
        _positive_int(n_samples)
        and values.ndim == 1
        and values.shape[0] >= 0
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_samples)
        and np.array_equal(values, np.unique(values))
    )


def _aligned_oob_prediction_inputs(
    prediction_blocks: PredictionBlockTuple,
    unsampled_index_blocks: IndexTuple,
    n_samples: int,
    prediction_width: int,
    n_outputs: int,
) -> bool:
    if not (
        _positive_int(n_samples)
        and _positive_int(prediction_width)
        and _positive_int(n_outputs)
        and len(prediction_blocks) >= 1
        and len(prediction_blocks) == len(unsampled_index_blocks)
    ):
        return False
    for block, indices in zip(prediction_blocks, unsampled_index_blocks):
        if not (
            _prediction_block_valid(block, expected_width=prediction_width, n_outputs=n_outputs)
            and _unsampled_index_block_valid(indices, n_samples)
        ):
            return False
        if np.asarray(block, dtype=np.float64).shape[0] != np.asarray(indices, dtype=np.int64).shape[0]:
            return False
    return True


def _prediction_totals_valid(
    values: object,
    n_samples: int,
    prediction_width: int,
    n_outputs: int,
) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.shape == (n_samples, prediction_width, n_outputs)
        and np.all(np.isfinite(array))
    )


def _prediction_counts_valid(values: object, n_samples: int, n_outputs: int) -> bool:
    array = np.asarray(values)
    return bool(
        array.shape == (n_samples, n_outputs)
        and np.issubdtype(array.dtype, np.integer)
        and np.all(array >= 0)
    )


def _counts_match_totals(prediction_totals: object, prediction_counts: object) -> bool:
    totals = np.asarray(prediction_totals, dtype=np.float64)
    counts = np.asarray(prediction_counts)
    return bool(totals.ndim == 3 and counts.shape == (totals.shape[0], totals.shape[2]))


def _uncovered_mask_valid(result: object, prediction_counts: object) -> bool:
    values = np.asarray(result)
    counts = np.asarray(prediction_counts)
    return bool(values.shape == (counts.shape[0],) and values.dtype == np.bool_)


def _averaged_predictions_valid(result: object, prediction_totals: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    totals = np.asarray(prediction_totals, dtype=np.float64)
    return bool(values.shape == totals.shape and np.all(np.isfinite(values)))


@register_atom(witness_forest_classifier_oob_prediction_block)
@icontract.require(
    lambda prediction: _classifier_prediction_valid(prediction),
    "prediction must be a finite normalized classifier probability matrix or rank-3 tensor",
)
@icontract.ensure(
    lambda result: np.asarray(result, dtype=np.float64).ndim == 3,
    "classifier OOB prediction block must be rank-3",
)
def forest_classifier_oob_prediction_block(
    prediction: NDArray[np.float64],
) -> PredictionBlock:
    """Format one tree's classifier OOB predictions the way sklearn stores them during accumulation."""
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim == 2:
        return np.asarray(values[..., np.newaxis], dtype=np.float64)
    return np.asarray(np.rollaxis(values, axis=0, start=3), dtype=np.float64)


@register_atom(witness_forest_regressor_oob_prediction_block)
@icontract.require(
    lambda prediction: _regressor_prediction_valid(prediction),
    "prediction must be a finite regressor output vector or matrix",
)
@icontract.ensure(
    lambda result: np.asarray(result, dtype=np.float64).ndim == 3,
    "regressor OOB prediction block must be rank-3",
)
def forest_regressor_oob_prediction_block(
    prediction: NDArray[np.float64],
) -> PredictionBlock:
    """Format one tree's regressor OOB predictions the way sklearn stores them during accumulation."""
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim == 1:
        return np.asarray(values[:, np.newaxis, np.newaxis], dtype=np.float64)
    return np.asarray(values[:, np.newaxis, :], dtype=np.float64)


@register_atom(witness_forest_oob_prediction_totals)
@icontract.require(
    lambda prediction_blocks, unsampled_index_blocks, n_samples, prediction_width, n_outputs: _aligned_oob_prediction_inputs(
        prediction_blocks,
        unsampled_index_blocks,
        n_samples,
        prediction_width,
        n_outputs,
    ),
    "prediction blocks and unsampled index blocks must align with sklearn forest OOB accumulation",
)
@icontract.ensure(
    lambda result, n_samples, prediction_width, n_outputs: _prediction_totals_valid(
        result,
        n_samples,
        prediction_width,
        n_outputs,
    ),
    "OOB prediction totals must be a finite sample-by-width-by-output tensor",
)
def forest_oob_prediction_totals(
    prediction_blocks: PredictionBlockTuple,
    unsampled_index_blocks: IndexTuple,
    *,
    n_samples: int,
    prediction_width: int,
    n_outputs: int,
) -> PredictionBlock:
    """Accumulate forest OOB prediction blocks into sklearn's sample-by-width-by-output tensor."""
    totals = np.zeros((n_samples, prediction_width, n_outputs), dtype=np.float64)
    for block, unsampled_indices in zip(prediction_blocks, unsampled_index_blocks):
        rows = np.asarray(unsampled_indices, dtype=np.int64)
        totals[rows, ...] += np.asarray(block, dtype=np.float64)
    return np.asarray(totals, dtype=np.float64)


@register_atom(witness_forest_oob_prediction_counts)
@icontract.require(
    lambda unsampled_index_blocks, n_samples, n_outputs: (
        _positive_int(n_samples)
        and _positive_int(n_outputs)
        and len(unsampled_index_blocks) >= 1
        and all(_unsampled_index_block_valid(block, n_samples) for block in unsampled_index_blocks)
    ),
    "unsampled index blocks must be a nonempty tuple of valid unique in-range index vectors",
)
@icontract.ensure(
    lambda result, n_samples, n_outputs: _prediction_counts_valid(result, n_samples, n_outputs),
    "OOB prediction counts must be a nonnegative sample-by-output integer matrix",
)
def forest_oob_prediction_counts(
    unsampled_index_blocks: IndexTuple,
    *,
    n_samples: int,
    n_outputs: int,
) -> NDArray[np.int64]:
    """Count how many forest OOB predictions contribute to each sample and output."""
    counts = np.zeros((n_samples, n_outputs), dtype=np.int64)
    for block in unsampled_index_blocks:
        counts[np.asarray(block, dtype=np.int64), :] += 1
    return np.asarray(counts, dtype=np.int64)


@register_atom(witness_forest_oob_uncovered_mask)
@icontract.require(
    lambda prediction_counts: _prediction_counts_valid(
        prediction_counts,
        np.asarray(prediction_counts).shape[0],
        np.asarray(prediction_counts).shape[1],
    ),
    "prediction_counts must be a nonnegative sample-by-output integer matrix",
)
@icontract.ensure(
    lambda result, prediction_counts: _uncovered_mask_valid(result, prediction_counts),
    "uncovered mask must have one boolean entry per sample",
)
def forest_oob_uncovered_mask(
    prediction_counts: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """Mark samples that receive no forest OOB predictions."""
    counts = np.asarray(prediction_counts, dtype=np.int64)
    return np.asarray(np.all(counts == 0, axis=1), dtype=np.bool_)


@register_atom(witness_forest_oob_average_predictions)
@icontract.require(
    lambda prediction_totals, prediction_counts: _counts_match_totals(prediction_totals, prediction_counts),
    "prediction_totals and prediction_counts must agree on sample and output axes",
)
@icontract.require(
    lambda prediction_totals: _prediction_totals_valid(
        prediction_totals,
        np.asarray(prediction_totals, dtype=np.float64).shape[0],
        np.asarray(prediction_totals, dtype=np.float64).shape[1],
        np.asarray(prediction_totals, dtype=np.float64).shape[2],
    ),
    "prediction_totals must be a finite sample-by-width-by-output tensor",
)
@icontract.require(
    lambda prediction_counts: _prediction_counts_valid(
        prediction_counts,
        np.asarray(prediction_counts).shape[0],
        np.asarray(prediction_counts).shape[1],
    ),
    "prediction_counts must be a nonnegative sample-by-output integer matrix",
)
@icontract.ensure(
    lambda result, prediction_totals: _averaged_predictions_valid(result, prediction_totals),
    "averaged OOB predictions must preserve the tensor shape and remain finite",
)
def forest_oob_average_predictions(
    prediction_totals: PredictionBlock,
    prediction_counts: NDArray[np.int64],
) -> PredictionBlock:
    """Average forest OOB prediction totals with sklearn's zero-count safeguard."""
    totals = np.asarray(prediction_totals, dtype=np.float64).copy()
    counts = np.asarray(prediction_counts, dtype=np.int64).copy()
    counts[counts == 0] = 1
    for output_index in range(totals.shape[2]):
        totals[..., output_index] /= counts[:, [output_index]]
    return np.asarray(totals, dtype=np.float64)
