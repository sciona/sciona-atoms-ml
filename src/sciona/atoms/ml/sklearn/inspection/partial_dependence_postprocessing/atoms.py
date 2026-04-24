"""Partial-dependence brute postprocessing atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_assign_grid_values,
    witness_partial_dependence_average_response_sequence,
    witness_partial_dependence_finalize_averages,
    witness_partial_dependence_finalize_predictions,
    witness_partial_dependence_stack_response_sequence,
)

TaskKind = Literal["regression", "classification"]


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_response_array(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim in (1, 2)
        and array.shape[0] >= 1
        and (array.ndim == 1 or array.shape[1] >= 1)
        and np.all(np.isfinite(array))
    )


def _feature_tuple_valid(features: tuple[int, ...], n_features: int) -> bool:
    return bool(
        isinstance(features, tuple)
        and len(features) >= 1
        and len(set(features)) == len(features)
        and all(isinstance(feature, int) and not isinstance(feature, bool) and 0 <= feature < n_features for feature in features)
    )


def _response_sequence_valid(responses: Sequence[object]) -> bool:
    if not isinstance(responses, tuple) or len(responses) < 1:
        return False
    arrays: list[NDArray[np.float64]] = []
    for response in responses:
        if not _finite_response_array(response):
            return False
        arrays.append(np.asarray(response, dtype=np.float64))
    shape = arrays[0].shape
    return bool(all(array.shape == shape for array in arrays))


def _sample_weight_valid(sample_weight: NDArray[np.float64] | None, responses: Sequence[object]) -> bool:
    if sample_weight is None:
        return True
    if not _finite_vector(sample_weight) or not _response_sequence_valid(responses):
        return False
    weights = np.asarray(sample_weight, dtype=np.float64)
    first = np.asarray(responses[0], dtype=np.float64)
    return bool(weights.shape[0] == first.shape[0])


def _average_stack_valid(result: NDArray[np.float64], responses: Sequence[object]) -> bool:
    if not _response_sequence_valid(responses):
        return False
    values = np.asarray(result, dtype=np.float64)
    first = np.asarray(responses[0], dtype=np.float64)
    expected_shape = (len(responses),) if first.ndim == 1 else (first.shape[1], len(responses))
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _response_stack_valid(result: NDArray[np.float64], responses: Sequence[object]) -> bool:
    if not _response_sequence_valid(responses):
        return False
    values = np.asarray(result, dtype=np.float64)
    first = np.asarray(responses[0], dtype=np.float64)
    expected_shape = (first.shape[0], len(responses)) if first.ndim == 1 else (first.shape[1], first.shape[0], len(responses))
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _task_kind_valid(task_kind: str) -> bool:
    return task_kind in {"regression", "classification"}


def _stacked_predictions_valid(stacked_predictions: object, n_samples: int) -> bool:
    try:
        values = np.asarray(stacked_predictions, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
        return False
    if values.ndim == 2:
        return bool(values.shape[0] == n_samples and values.shape[1] >= 1 and np.all(np.isfinite(values)))
    if values.ndim == 3:
        return bool(values.shape[0] >= 1 and values.shape[1] == n_samples and values.shape[2] >= 1 and np.all(np.isfinite(values)))
    return False


def _stacked_averages_valid(stacked_averages: object) -> bool:
    try:
        values = np.asarray(stacked_averages, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if values.ndim == 1:
        return bool(values.shape[0] >= 1 and np.all(np.isfinite(values)))
    if values.ndim == 2:
        return bool(values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))
    return False


def _finalized_predictions_valid(
    result: NDArray[np.float64],
    stacked_predictions: NDArray[np.float64],
    task_kind: TaskKind,
    n_samples: int,
) -> bool:
    if not _stacked_predictions_valid(stacked_predictions, n_samples):
        return False
    values = np.asarray(result, dtype=np.float64)
    stacked = np.asarray(stacked_predictions, dtype=np.float64)
    if task_kind == "regression" and stacked.ndim == 2:
        expected_shape = (n_samples, stacked.shape[1])
    elif task_kind == "classification" and stacked.ndim == 3 and stacked.shape[0] == 2:
        expected_shape = (n_samples, stacked.shape[2])
    else:
        expected_shape = stacked.shape
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _finalized_averages_valid(
    result: NDArray[np.float64],
    stacked_averages: NDArray[np.float64],
    task_kind: TaskKind,
) -> bool:
    if not _stacked_averages_valid(stacked_averages):
        return False
    values = np.asarray(result, dtype=np.float64)
    stacked = np.asarray(stacked_averages, dtype=np.float64)
    if task_kind == "regression" and stacked.ndim == 1:
        expected_shape = (1, stacked.shape[0])
    elif task_kind == "classification" and stacked.ndim == 2 and stacked.shape[0] == 2:
        expected_shape = (1, stacked.shape[1])
    else:
        expected_shape = stacked.shape
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


@register_atom(witness_partial_dependence_assign_grid_values)
@icontract.require(lambda X: _finite_matrix(X), "X must be a nonempty finite matrix")
@icontract.require(lambda new_values: _finite_vector(new_values), "new_values must be a nonempty finite vector")
@icontract.require(
    lambda X, new_values, features: _feature_tuple_valid(features, np.asarray(X, dtype=np.float64).shape[1])
    and len(features) == np.asarray(new_values, dtype=np.float64).shape[0],
    "features must be a unique tuple of valid feature indices matching new_values",
)
@icontract.ensure(lambda result, X: _finite_matrix(result) and np.asarray(result).shape == np.asarray(X).shape, "assigned matrix must stay finite and preserve shape")
def partial_dependence_assign_grid_values(
    X: NDArray[np.float64],
    new_values: NDArray[np.float64],
    *,
    features: tuple[int, ...],
) -> NDArray[np.float64]:
    """Assign one grid point into the requested dense feature columns."""
    updated = np.array(X, dtype=np.float64, copy=True)
    values = np.asarray(new_values, dtype=np.float64)
    for index, feature in enumerate(features):
        updated[:, int(feature)] = values[index]
    return np.asarray(updated, dtype=np.float64)


@register_atom(witness_partial_dependence_average_response_sequence)
@icontract.require(lambda responses: _response_sequence_valid(responses), "responses must be a nonempty tuple of same-shaped finite 1D or 2D arrays")
@icontract.require(lambda responses, sample_weight=None: _sample_weight_valid(sample_weight, responses), "sample_weight must be None or a finite vector aligned with the sample axis")
@icontract.ensure(lambda result, responses: _average_stack_valid(result, responses), "averaged responses must match sklearn's stacked average shape")
def partial_dependence_average_response_sequence(
    responses: tuple[NDArray[np.float64], ...],
    *,
    sample_weight: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Average each supplied response array over samples and stack the results."""
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    averages = [
        np.average(np.asarray(response, dtype=np.float64), axis=0, weights=weights)
        for response in responses
    ]
    return np.asarray(np.array(averages).T, dtype=np.float64)


@register_atom(witness_partial_dependence_stack_response_sequence)
@icontract.require(lambda responses: _response_sequence_valid(responses), "responses must be a nonempty tuple of same-shaped finite 1D or 2D arrays")
@icontract.ensure(lambda result, responses: _response_stack_valid(result, responses), "stacked responses must match sklearn's transposed prediction stack shape")
def partial_dependence_stack_response_sequence(
    responses: tuple[NDArray[np.float64], ...],
) -> NDArray[np.float64]:
    """Stack a response sequence using sklearn's transpose convention."""
    arrays = tuple(np.asarray(response, dtype=np.float64) for response in responses)
    return np.asarray(np.array(arrays).T, dtype=np.float64)


@register_atom(witness_partial_dependence_finalize_predictions)
@icontract.require(lambda stacked_predictions, n_samples: _stacked_predictions_valid(stacked_predictions, n_samples), "stacked_predictions must be a finite 2D or 3D array aligned with n_samples")
@icontract.require(lambda task_kind: _task_kind_valid(task_kind), "task_kind must be 'regression' or 'classification'")
@icontract.ensure(lambda result, stacked_predictions, task_kind, n_samples: _finalized_predictions_valid(result, stacked_predictions, task_kind, n_samples), "finalized predictions must match sklearn's regression or classification output shape")
def partial_dependence_finalize_predictions(
    stacked_predictions: NDArray[np.float64],
    *,
    task_kind: TaskKind,
    n_samples: int,
) -> NDArray[np.float64]:
    """Apply sklearn's final prediction reshaping for regression or classification."""
    predictions = np.asarray(stacked_predictions, dtype=np.float64)
    if task_kind == "regression" and predictions.ndim == 2:
        return np.asarray(predictions.reshape(int(n_samples), -1), dtype=np.float64)
    if task_kind == "classification" and predictions.ndim == 3 and predictions.shape[0] == 2:
        return np.asarray(predictions[1].reshape(int(n_samples), -1), dtype=np.float64)
    return np.asarray(predictions, dtype=np.float64)


@register_atom(witness_partial_dependence_finalize_averages)
@icontract.require(lambda stacked_averages: _stacked_averages_valid(stacked_averages), "stacked_averages must be a finite 1D or 2D array")
@icontract.require(lambda task_kind: _task_kind_valid(task_kind), "task_kind must be 'regression' or 'classification'")
@icontract.ensure(lambda result, stacked_averages, task_kind: _finalized_averages_valid(result, stacked_averages, task_kind), "finalized averages must match sklearn's regression or classification output shape")
def partial_dependence_finalize_averages(
    stacked_averages: NDArray[np.float64],
    *,
    task_kind: TaskKind,
) -> NDArray[np.float64]:
    """Apply sklearn's final average-response reshaping for regression or classification."""
    averages = np.asarray(stacked_averages, dtype=np.float64)
    if task_kind == "regression" and averages.ndim == 1:
        return np.asarray(averages.reshape(1, -1), dtype=np.float64)
    if task_kind == "classification" and averages.ndim == 2 and averages.shape[0] == 2:
        return np.asarray(averages[1].reshape(1, -1), dtype=np.float64)
    return np.asarray(averages, dtype=np.float64)
