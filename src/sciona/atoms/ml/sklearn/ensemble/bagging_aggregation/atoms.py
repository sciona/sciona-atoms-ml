"""Bagging aggregation helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bagging_classifier_average_decision_function,
    witness_bagging_classifier_average_log_probabilities,
    witness_bagging_classifier_average_probabilities,
    witness_bagging_regressor_average_predictions,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _finite_probability_block(block: object) -> bool:
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


def _finite_log_probability_block(block: object) -> bool:
    try:
        values = np.asarray(block, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    finite = np.isfinite(values)
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.all(np.isfinite(values) | np.isneginf(values))
        and np.all(values[finite] <= 0.0)
        and np.allclose(np.sum(np.exp(values), axis=1), 1.0)
    )


def _class_index_block_valid(block: object, n_classes: int) -> bool:
    values = np.asarray(block)
    return bool(
        values.ndim == 1
        and values.shape[0] >= 1
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_classes)
        and np.unique(values).shape[0] == values.shape[0]
    )


def _aligned_probability_blocks(
    probability_blocks: tuple[NDArray[np.float64], ...],
    class_index_blocks: tuple[NDArray[np.int64], ...],
    n_classes: int,
    *,
    log_space: bool = False,
) -> bool:
    if not (_positive_int(n_classes) and len(probability_blocks) >= 1 and len(probability_blocks) == len(class_index_blocks)):
        return False
    n_samples: int | None = None
    validator = _finite_log_probability_block if log_space else _finite_probability_block
    for block, class_indices in zip(probability_blocks, class_index_blocks):
        if not validator(block):
            return False
        if not _class_index_block_valid(class_indices, n_classes):
            return False
        values = np.asarray(block, dtype=np.float64)
        indices = np.asarray(class_indices, dtype=np.int64)
        if values.shape[1] != indices.shape[0]:
            return False
        if n_samples is None:
            n_samples = int(values.shape[0])
        elif int(values.shape[0]) != n_samples:
            return False
    return True


def _average_probabilities_valid(result: NDArray[np.float64], probability_blocks: tuple[NDArray[np.float64], ...], n_classes: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_samples = np.asarray(probability_blocks[0], dtype=np.float64).shape[0]
    return bool(
        values.shape == (n_samples, n_classes)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _average_log_probabilities_valid(result: NDArray[np.float64], log_probability_blocks: tuple[NDArray[np.float64], ...], n_classes: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_samples = np.asarray(log_probability_blocks[0], dtype=np.float64).shape[0]
    return bool(
        values.shape == (n_samples, n_classes)
        and np.all(np.isfinite(values) | np.isneginf(values))
        and np.allclose(np.sum(np.exp(values), axis=1), 1.0)
    )


def _finite_same_shape_blocks(blocks: tuple[NDArray[np.float64], ...]) -> bool:
    if len(blocks) < 1:
        return False
    try:
        first = np.asarray(blocks[0], dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if first.ndim not in {1, 2} or any(dim < 1 for dim in first.shape) or not np.all(np.isfinite(first)):
        return False
    shape = first.shape
    for block in blocks[1:]:
        try:
            values = np.asarray(block, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if values.shape != shape or not np.all(np.isfinite(values)):
            return False
    return True


def _same_shape_result_valid(result: NDArray[np.float64], blocks: tuple[NDArray[np.float64], ...]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    expected_shape = np.asarray(blocks[0], dtype=np.float64).shape
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _aggregate_probability_blocks(
    probability_blocks: tuple[NDArray[np.float64], ...],
    class_index_blocks: tuple[NDArray[np.int64], ...],
    n_classes: int,
) -> NDArray[np.float64]:
    n_samples = np.asarray(probability_blocks[0], dtype=np.float64).shape[0]
    total = np.zeros((n_samples, n_classes), dtype=np.float64)
    for block, class_indices in zip(probability_blocks, class_index_blocks):
        total[:, np.asarray(class_indices, dtype=np.int64)] += np.asarray(block, dtype=np.float64)
    return total


@register_atom(witness_bagging_classifier_average_probabilities)
@icontract.require(lambda probability_blocks, class_index_blocks, n_classes: _aligned_probability_blocks(probability_blocks, class_index_blocks, n_classes), "probability blocks and class-index blocks must be aligned, normalized, nonempty, and share a sample count")
@icontract.ensure(lambda result, probability_blocks, n_classes: _average_probabilities_valid(result, probability_blocks, n_classes), "averaged probabilities must stay normalized per sample")
def bagging_classifier_average_probabilities(
    probability_blocks: tuple[NDArray[np.float64], ...],
    class_index_blocks: tuple[NDArray[np.int64], ...],
    *,
    n_classes: int,
) -> NDArray[np.float64]:
    """Average bagging classifier probability outputs with sklearn-style class alignment."""
    total = _aggregate_probability_blocks(probability_blocks, class_index_blocks, int(n_classes))
    return np.asarray(total / len(probability_blocks), dtype=np.float64)


@register_atom(witness_bagging_classifier_average_log_probabilities)
@icontract.require(lambda log_probability_blocks, class_index_blocks, n_classes: _aligned_probability_blocks(log_probability_blocks, class_index_blocks, n_classes, log_space=True), "log-probability blocks and class-index blocks must be aligned, nonempty, and share a sample count")
@icontract.ensure(lambda result, log_probability_blocks, n_classes: _average_log_probabilities_valid(result, log_probability_blocks, n_classes), "averaged log probabilities must preserve the aligned sample-by-class shape")
def bagging_classifier_average_log_probabilities(
    log_probability_blocks: tuple[NDArray[np.float64], ...],
    class_index_blocks: tuple[NDArray[np.int64], ...],
    *,
    n_classes: int,
) -> NDArray[np.float64]:
    """Average bagging classifier log probabilities with sklearn-style class alignment."""
    n_samples = np.asarray(log_probability_blocks[0], dtype=np.float64).shape[0]
    log_proba = np.empty((n_samples, int(n_classes)), dtype=np.float64)
    log_proba.fill(-np.inf)
    for block, class_indices in zip(log_probability_blocks, class_index_blocks):
        idx = np.asarray(class_indices, dtype=np.int64)
        log_proba[:, idx] = np.logaddexp(log_proba[:, idx], np.asarray(block, dtype=np.float64))
    log_proba -= np.log(len(log_probability_blocks))
    return np.asarray(log_proba, dtype=np.float64)


@register_atom(witness_bagging_classifier_average_decision_function)
@icontract.require(lambda decision_blocks: _finite_same_shape_blocks(decision_blocks), "decision blocks must be a nonempty tuple of finite arrays with matching shape")
@icontract.ensure(lambda result, decision_blocks: _same_shape_result_valid(result, decision_blocks), "averaged decision function must preserve the shared decision shape")
def bagging_classifier_average_decision_function(
    decision_blocks: tuple[NDArray[np.float64], ...],
) -> NDArray[np.float64]:
    """Average bagging classifier decision-function outputs."""
    return np.asarray(np.mean(np.stack([np.asarray(block, dtype=np.float64) for block in decision_blocks], axis=0), axis=0), dtype=np.float64)


@register_atom(witness_bagging_regressor_average_predictions)
@icontract.require(lambda prediction_blocks: _finite_same_shape_blocks(prediction_blocks), "prediction blocks must be a nonempty tuple of finite arrays with matching shape")
@icontract.ensure(lambda result, prediction_blocks: _same_shape_result_valid(result, prediction_blocks), "averaged predictions must preserve the shared prediction shape")
def bagging_regressor_average_predictions(
    prediction_blocks: tuple[NDArray[np.float64], ...],
) -> NDArray[np.float64]:
    """Average bagging regressor predictions."""
    return np.asarray(np.mean(np.stack([np.asarray(block, dtype=np.float64) for block in prediction_blocks], axis=0), axis=0), dtype=np.float64)
