"""MiniBatchDictionaryLearning fit-prelude atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dictionary_learning_minibatch_monitor_state,
    witness_dictionary_learning_minibatch_old_dictionary,
    witness_dictionary_learning_minibatch_training_data,
    witness_dictionary_learning_minibatch_verbose_message,
)

Matrix = NDArray[np.floating]


def _finite_matrix(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.issubdtype(array.dtype, np.number)
        and np.all(np.isfinite(array))
    )


def _bool_int(value: object) -> bool:
    return isinstance(value, (bool, int)) and not isinstance(value, np.ndarray)


def _optional_permutation_valid(X: object, shuffle: object, permutation: object) -> bool:
    if not (_finite_matrix(X) and isinstance(shuffle, bool)):
        return False
    if not shuffle:
        return permutation is None
    try:
        indices = np.asarray(permutation, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    n_samples = np.asarray(X).shape[0]
    return bool(
        indices.shape == (n_samples,)
        and np.array_equal(np.sort(indices), np.arange(n_samples, dtype=np.int64))
    )


def _training_data_valid(result: object, X: object, shuffle: bool, permutation: object | None) -> bool:
    if not _finite_matrix(result):
        return False
    values = np.asarray(result)
    X_values = np.asarray(X)
    if values.shape != X_values.shape or values.dtype != X_values.dtype:
        return False
    if shuffle:
        indices = np.asarray(permutation, dtype=np.int64)
        return np.array_equal(values, X_values[indices])
    return np.array_equal(values, X_values)


def _old_dictionary_valid(result: object, dictionary: object) -> bool:
    if not (_finite_matrix(result) and _finite_matrix(dictionary)):
        return False
    result_values = np.asarray(result)
    dictionary_values = np.asarray(dictionary)
    return bool(
        result_values.shape == dictionary_values.shape
        and result_values.dtype == dictionary_values.dtype
        and np.array_equal(result_values, dictionary_values)
    )


def _verbose_message_valid(result: object, verbose: bool | int) -> bool:
    if bool(verbose):
        return result == "[dict_learning]"
    return result is None


def _monitor_state_valid(result: object) -> bool:
    return isinstance(result, tuple) and result == (None, None, 0)


@register_atom(witness_dictionary_learning_minibatch_training_data)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite numeric matrix")
@icontract.require(lambda shuffle: isinstance(shuffle, bool), "shuffle must be boolean")
@icontract.require(
    lambda X, shuffle, permutation=None: _optional_permutation_valid(X, shuffle, permutation),
    "permutation must be None for shuffle=False or a full row permutation for shuffle=True",
)
@icontract.ensure(
    lambda result, X, shuffle, permutation=None: _training_data_valid(result, X, shuffle, permutation),
    "result must match sklearn's shuffled-or-original training matrix selection",
)
def dictionary_learning_minibatch_training_data(
    X: Matrix,
    shuffle: bool,
    permutation: NDArray[np.int64] | None = None,
) -> Matrix:
    """Choose sklearn's minibatch training matrix before the loop."""
    values = np.asarray(X)
    if not shuffle:
        return values
    assert permutation is not None
    return np.asarray(values[np.asarray(permutation, dtype=np.int64)])


@register_atom(witness_dictionary_learning_minibatch_old_dictionary)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite numeric matrix")
@icontract.ensure(
    lambda result, dictionary: _old_dictionary_valid(result, dictionary),
    "result must be a value-equal dictionary copy",
)
def dictionary_learning_minibatch_old_dictionary(
    dictionary: Matrix,
) -> Matrix:
    """Copy the initial dictionary like MiniBatchDictionaryLearning.fit."""
    return np.asarray(dictionary).copy()


@register_atom(witness_dictionary_learning_minibatch_verbose_message)
@icontract.require(lambda verbose: _bool_int(verbose), "verbose must be a bool or int")
@icontract.ensure(lambda result, verbose: _verbose_message_valid(result, verbose), "result must be the optional sklearn verbose banner")
def dictionary_learning_minibatch_verbose_message(
    verbose: bool | int,
) -> str | None:
    """Emit sklearn's fixed dictionary-learning verbose banner when enabled."""
    if bool(verbose):
        return "[dict_learning]"
    return None


@register_atom(witness_dictionary_learning_minibatch_monitor_state)
@icontract.require(
    lambda max_no_improvement: max_no_improvement is None
    or (isinstance(max_no_improvement, int) and not isinstance(max_no_improvement, bool) and max_no_improvement >= 0),
    "max_no_improvement must be None or a nonnegative integer",
)
@icontract.ensure(lambda result: _monitor_state_valid(result), "result must be sklearn's initial monitor state")
def dictionary_learning_minibatch_monitor_state(
    max_no_improvement: int | None,
) -> tuple[None, None, int]:
    """Initialize sklearn's convergence-monitor bookkeeping fields."""
    del max_no_improvement
    return None, None, 0
