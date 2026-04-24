"""Bagging out-of-bag aggregation helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bagging_classifier_oob_decision_function,
    witness_bagging_classifier_oob_label_indices,
    witness_bagging_classifier_oob_probability_totals,
    witness_bagging_classifier_oob_vote_totals,
    witness_bagging_regressor_oob_predictions,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _sample_index_block_valid(block: object, n_samples: int) -> bool:
    values = np.asarray(block)
    return bool(
        _positive_int(n_samples)
        and values.ndim == 1
        and values.shape[0] >= 1
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_samples)
    )


def _oob_rows(sample_indices: NDArray[np.int64], n_samples: int) -> NDArray[np.int64]:
    in_bag_mask = np.zeros(n_samples, dtype=np.bool_)
    in_bag_mask[np.asarray(sample_indices, dtype=np.int64)] = True
    return np.asarray(np.flatnonzero(~in_bag_mask), dtype=np.int64)


def _probability_block_valid(block: object) -> bool:
    try:
        values = np.asarray(block, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _class_index_block_valid(block: object, n_classes: int) -> bool:
    values = np.asarray(block)
    return bool(
        _positive_int(n_classes)
        and values.ndim == 1
        and values.shape[0] >= 1
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_classes)
        and np.unique(values).shape[0] == values.shape[0]
    )


def _label_block_valid(block: object, n_classes: int) -> bool:
    values = np.asarray(block)
    return bool(
        _positive_int(n_classes)
        and values.ndim == 1
        and values.shape[0] >= 1
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_classes)
    )


def _regression_block_valid(block: object) -> bool:
    try:
        values = np.asarray(block, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)))


def _aligned_probability_oob_inputs(
    probability_blocks: tuple[NDArray[np.float64], ...],
    class_index_blocks: tuple[NDArray[np.int64], ...],
    sample_index_blocks: tuple[NDArray[np.int64], ...],
    n_samples: int,
    n_classes: int,
) -> bool:
    if not (
        _positive_int(n_samples)
        and _positive_int(n_classes)
        and len(probability_blocks) >= 1
        and len(probability_blocks) == len(class_index_blocks) == len(sample_index_blocks)
    ):
        return False
    for proba, class_indices, sample_indices in zip(probability_blocks, class_index_blocks, sample_index_blocks):
        if not (_probability_block_valid(proba) and _class_index_block_valid(class_indices, n_classes) and _sample_index_block_valid(sample_indices, n_samples)):
            return False
        rows = _oob_rows(np.asarray(sample_indices, dtype=np.int64), n_samples)
        if np.asarray(proba, dtype=np.float64).shape != (rows.shape[0], np.asarray(class_indices, dtype=np.int64).shape[0]):
            return False
    return True


def _aligned_vote_oob_inputs(
    predicted_label_blocks: tuple[NDArray[np.int64], ...],
    sample_index_blocks: tuple[NDArray[np.int64], ...],
    n_samples: int,
    n_classes: int,
) -> bool:
    if not (
        _positive_int(n_samples)
        and _positive_int(n_classes)
        and len(predicted_label_blocks) >= 1
        and len(predicted_label_blocks) == len(sample_index_blocks)
    ):
        return False
    for labels, sample_indices in zip(predicted_label_blocks, sample_index_blocks):
        if not (_label_block_valid(labels, n_classes) and _sample_index_block_valid(sample_indices, n_samples)):
            return False
        if np.asarray(labels, dtype=np.int64).shape != _oob_rows(np.asarray(sample_indices, dtype=np.int64), n_samples).shape:
            return False
    return True


def _aligned_regression_oob_inputs(
    prediction_blocks: tuple[NDArray[np.float64], ...],
    sample_index_blocks: tuple[NDArray[np.int64], ...],
    n_samples: int,
) -> bool:
    if not (_positive_int(n_samples) and len(prediction_blocks) >= 1 and len(prediction_blocks) == len(sample_index_blocks)):
        return False
    for preds, sample_indices in zip(prediction_blocks, sample_index_blocks):
        if not (_regression_block_valid(preds) and _sample_index_block_valid(sample_indices, n_samples)):
            return False
        if np.asarray(preds, dtype=np.float64).shape != _oob_rows(np.asarray(sample_indices, dtype=np.int64), n_samples).shape:
            return False
    return True


def _total_matrix_valid(result: NDArray[np.float64], n_samples: int, n_classes: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (n_samples, n_classes) and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _decision_function_valid(result: NDArray[np.float64], prediction_totals: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    totals = np.asarray(prediction_totals, dtype=np.float64)
    if values.shape != totals.shape:
        return False
    row_sums = np.sum(totals, axis=1)
    positive = row_sums > 0.0
    if np.any(positive):
        positive_rows = values[positive]
        if not (
            np.all(np.isfinite(positive_rows))
            and np.all(positive_rows >= 0.0)
            and np.allclose(np.sum(positive_rows, axis=1), 1.0)
        ):
            return False
    if np.any(~positive):
        return bool(np.all(np.isnan(values[~positive])))
    return True


def _label_indices_valid(result: NDArray[np.int64], prediction_totals: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    totals = np.asarray(prediction_totals, dtype=np.float64)
    return bool(
        values.ndim == 1
        and values.shape == (totals.shape[0],)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < totals.shape[1])
    )


def _regression_result_valid(result: NDArray[np.float64], n_samples: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (n_samples,) and np.all(np.isfinite(values)))


@register_atom(witness_bagging_classifier_oob_probability_totals)
@icontract.require(
    lambda probability_blocks, class_index_blocks, sample_index_blocks, n_samples, n_classes: _aligned_probability_oob_inputs(
        probability_blocks, class_index_blocks, sample_index_blocks, n_samples, n_classes
    ),
    "probability blocks, class-index blocks, and sample-index blocks must align with each estimator's out-of-bag rows",
)
@icontract.ensure(
    lambda result, n_samples, n_classes: _total_matrix_valid(result, n_samples, n_classes),
    "OOB probability totals must be a finite nonnegative sample-by-class matrix",
)
def bagging_classifier_oob_probability_totals(
    probability_blocks: tuple[NDArray[np.float64], ...],
    class_index_blocks: tuple[NDArray[np.int64], ...],
    sample_index_blocks: tuple[NDArray[np.int64], ...],
    *,
    n_samples: int,
    n_classes: int,
) -> NDArray[np.float64]:
    """Accumulate per-sample held-out class totals from probability outputs."""
    totals = np.zeros((n_samples, n_classes), dtype=np.float64)
    for proba, class_indices, sample_indices in zip(probability_blocks, class_index_blocks, sample_index_blocks):
        rows = _oob_rows(np.asarray(sample_indices, dtype=np.int64), n_samples)
        idx = np.asarray(class_indices, dtype=np.int64)
        totals[np.ix_(rows, idx)] += np.asarray(proba, dtype=np.float64)
    return np.asarray(totals, dtype=np.float64)


@register_atom(witness_bagging_classifier_oob_vote_totals)
@icontract.require(
    lambda predicted_label_blocks, sample_index_blocks, n_samples, n_classes: _aligned_vote_oob_inputs(
        predicted_label_blocks, sample_index_blocks, n_samples, n_classes
    ),
    "predicted-label blocks and sample-index blocks must align with each estimator's out-of-bag rows",
)
@icontract.ensure(
    lambda result, n_samples, n_classes: _total_matrix_valid(result, n_samples, n_classes),
    "OOB vote totals must be a finite nonnegative sample-by-class matrix",
)
def bagging_classifier_oob_vote_totals(
    predicted_label_blocks: tuple[NDArray[np.int64], ...],
    sample_index_blocks: tuple[NDArray[np.int64], ...],
    *,
    n_samples: int,
    n_classes: int,
) -> NDArray[np.float64]:
    """Accumulate sklearn BaggingClassifier OOB class totals from vote-only labels."""
    totals = np.zeros((n_samples, n_classes), dtype=np.float64)
    for labels, sample_indices in zip(predicted_label_blocks, sample_index_blocks):
        rows = _oob_rows(np.asarray(sample_indices, dtype=np.int64), n_samples)
        label_values = np.asarray(labels, dtype=np.int64)
        totals[rows, label_values] += 1.0
    return np.asarray(totals, dtype=np.float64)


@register_atom(witness_bagging_classifier_oob_decision_function)
@icontract.require(
    lambda prediction_totals: _total_matrix_valid(
        np.asarray(prediction_totals, dtype=np.float64),
        np.asarray(prediction_totals, dtype=np.float64).shape[0],
        np.asarray(prediction_totals, dtype=np.float64).shape[1],
    ),
    "prediction_totals must be a finite nonnegative sample-by-class matrix",
)
@icontract.ensure(
    lambda result, prediction_totals: _decision_function_valid(result, prediction_totals),
    "OOB decision function must normalize positive-total rows and leave zero-total rows as NaN",
)
def bagging_classifier_oob_decision_function(
    prediction_totals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize held-out class totals into per-sample class shares."""
    totals = np.asarray(prediction_totals, dtype=np.float64)
    row_sums = np.sum(totals, axis=1)[:, np.newaxis]
    with np.errstate(invalid="ignore", divide="ignore"):
        result = totals / row_sums
    return np.asarray(result, dtype=np.float64)


@register_atom(witness_bagging_classifier_oob_label_indices)
@icontract.require(
    lambda prediction_totals: _total_matrix_valid(
        np.asarray(prediction_totals, dtype=np.float64),
        np.asarray(prediction_totals, dtype=np.float64).shape[0],
        np.asarray(prediction_totals, dtype=np.float64).shape[1],
    ),
    "prediction_totals must be a finite nonnegative sample-by-class matrix",
)
@icontract.ensure(
    lambda result, prediction_totals: _label_indices_valid(result, prediction_totals),
    "OOB label indices must be integer argmax selections along the class axis",
)
def bagging_classifier_oob_label_indices(
    prediction_totals: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Return sklearn's OOB predicted-class indices from class totals."""
    return np.asarray(np.argmax(np.asarray(prediction_totals, dtype=np.float64), axis=1), dtype=np.int64)


@register_atom(witness_bagging_regressor_oob_predictions)
@icontract.require(
    lambda prediction_blocks, sample_index_blocks, n_samples: _aligned_regression_oob_inputs(
        prediction_blocks, sample_index_blocks, n_samples
    ),
    "prediction blocks and sample-index blocks must align with each estimator's out-of-bag rows",
)
@icontract.ensure(
    lambda result, n_samples: _regression_result_valid(result, n_samples),
    "OOB regression predictions must be a finite vector with one value per sample",
)
def bagging_regressor_oob_predictions(
    prediction_blocks: tuple[NDArray[np.float64], ...],
    sample_index_blocks: tuple[NDArray[np.int64], ...],
    *,
    n_samples: int,
) -> NDArray[np.float64]:
    """Average BaggingRegressor OOB predictions with sklearn's zero-count safeguard."""
    predictions = np.zeros((n_samples,), dtype=np.float64)
    n_predictions = np.zeros((n_samples,), dtype=np.float64)
    for preds, sample_indices in zip(prediction_blocks, sample_index_blocks):
        rows = _oob_rows(np.asarray(sample_indices, dtype=np.int64), n_samples)
        prediction_values = np.asarray(preds, dtype=np.float64)
        predictions[rows] += prediction_values
        n_predictions[rows] += 1.0
    n_predictions[n_predictions == 0.0] = 1.0
    return np.asarray(predictions / n_predictions, dtype=np.float64)
