"""One-vs-rest prediction-output helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_one_vs_rest_predict_argmaxima_init,
    witness_one_vs_rest_predict_labels_from_argmaxima,
    witness_one_vs_rest_predict_maxima_init,
    witness_one_vs_rest_predict_multiclass_update,
)

UpdateState = tuple[NDArray[np.float64], NDArray[np.int64]]


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _classes_valid(classes: object) -> bool:
    try:
        values = np.asarray(classes, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 1
        and values.shape[0] >= 1
        and np.all(np.isfinite(values))
        and np.unique(values).shape[0] == values.shape[0]
    )


def _finite_score_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _maxima_valid(maxima: object) -> bool:
    try:
        values = np.asarray(maxima, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values) | np.isneginf(values)))


def _argmaxima_valid(argmaxima: object, n_samples: int) -> bool:
    values = np.asarray(argmaxima)
    return bool(
        values.ndim == 1
        and values.shape == (n_samples,)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
    )


def _update_inputs_valid(maxima: object, argmaxima: object, pred: object, class_index: object) -> bool:
    if not _maxima_valid(maxima) or not _finite_score_vector(pred):
        return False
    maxima_values = np.asarray(maxima, dtype=np.float64)
    pred_values = np.asarray(pred, dtype=np.float64)
    return bool(
        _argmaxima_valid(argmaxima, pred_values.shape[0])
        and maxima_values.shape == pred_values.shape
        and isinstance(class_index, int)
        and not isinstance(class_index, bool)
        and class_index >= 0
    )


def _update_result_valid(result: object, maxima: object, argmaxima: object, pred: object, class_index: int) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    next_maxima, next_argmaxima = result
    if not (_maxima_valid(next_maxima) and _argmaxima_valid(next_argmaxima, np.asarray(pred, dtype=np.float64).shape[0])):
        return False
    prev_maxima = np.asarray(maxima, dtype=np.float64)
    prev_argmaxima = np.asarray(argmaxima, dtype=np.int64)
    pred_values = np.asarray(pred, dtype=np.float64)
    next_maxima_values = np.asarray(next_maxima, dtype=np.float64)
    next_argmaxima_values = np.asarray(next_argmaxima, dtype=np.int64)
    expected_maxima = np.maximum(prev_maxima, pred_values)
    expected_argmaxima = prev_argmaxima.copy()
    expected_argmaxima[expected_maxima == pred_values] = class_index
    return bool(
        np.array_equal(next_maxima_values, expected_maxima)
        and np.array_equal(next_argmaxima_values, expected_argmaxima)
    )


def _labels_valid(result: object, n_samples: int, classes: object) -> bool:
    try:
        values = np.asarray(result, dtype=np.float64)
        class_values = np.asarray(classes, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.shape == (n_samples,) and np.all(np.isin(values, class_values)))


@register_atom(witness_one_vs_rest_predict_maxima_init)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result, n_samples: _maxima_valid(result) and np.asarray(result).shape == (n_samples,) and np.all(np.isneginf(np.asarray(result, dtype=np.float64))), "maxima must be a -inf-filled vector")
def one_vs_rest_predict_maxima_init(n_samples: int) -> NDArray[np.float64]:
    """Initialize sklearn's running maxima vector for multiclass OvR predict."""
    maxima = np.empty(int(n_samples), dtype=np.float64)
    maxima.fill(-np.inf)
    return np.asarray(maxima, dtype=np.float64)


@register_atom(witness_one_vs_rest_predict_argmaxima_init)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result, n_samples: _argmaxima_valid(result, n_samples) and np.all(np.asarray(result, dtype=np.int64) == 0), "argmaxima must be a zero-filled integer vector")
def one_vs_rest_predict_argmaxima_init(n_samples: int) -> NDArray[np.int64]:
    """Initialize sklearn's running winning-class index vector for multiclass OvR predict."""
    return np.zeros(int(n_samples), dtype=np.int64)


@register_atom(witness_one_vs_rest_predict_multiclass_update)
@icontract.require(
    lambda maxima, argmaxima, pred, class_index: _update_inputs_valid(maxima, argmaxima, pred, class_index),
    "maxima, argmaxima, pred, and class_index must be compatible",
)
@icontract.ensure(
    lambda result, maxima, argmaxima, pred, class_index: _update_result_valid(result, maxima, argmaxima, pred, class_index),
    "updated state must follow sklearn's running-maximum and later-tie-wins rule",
)
def one_vs_rest_predict_multiclass_update(
    maxima: NDArray[np.float64],
    argmaxima: NDArray[np.int64],
    pred: NDArray[np.float64],
    *,
    class_index: int,
) -> UpdateState:
    """Update sklearn's running multiclass OvR predict state for one estimator output."""
    maxima_values = np.asarray(maxima, dtype=np.float64).copy()
    argmaxima_values = np.asarray(argmaxima, dtype=np.int64).copy()
    pred_values = np.asarray(pred, dtype=np.float64)
    np.maximum(maxima_values, pred_values, out=maxima_values)
    argmaxima_values[maxima_values == pred_values] = int(class_index)
    return np.asarray(maxima_values, dtype=np.float64), np.asarray(argmaxima_values, dtype=np.int64)


@register_atom(witness_one_vs_rest_predict_labels_from_argmaxima)
@icontract.require(lambda argmaxima: _argmaxima_valid(argmaxima, np.asarray(argmaxima).shape[0]), "argmaxima must be a nonempty integer vector")
@icontract.require(lambda classes: _classes_valid(classes), "classes must be a finite unique class vector")
@icontract.require(lambda argmaxima, classes: np.all(np.asarray(argmaxima, dtype=np.int64) < np.asarray(classes, dtype=np.float64).shape[0]), "argmaxima must index into classes")
@icontract.ensure(lambda result, argmaxima, classes: _labels_valid(result, np.asarray(argmaxima).shape[0], classes), "labels must come from classes")
def one_vs_rest_predict_labels_from_argmaxima(
    argmaxima: NDArray[np.int64],
    classes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Decode sklearn's running winning-class indices into multiclass OvR labels."""
    argmaxima_values = np.asarray(argmaxima, dtype=np.int64)
    class_values = np.asarray(classes, dtype=np.float64)
    return np.asarray(class_values[argmaxima_values], dtype=np.float64)
