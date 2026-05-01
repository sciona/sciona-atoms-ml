"""MiniBatchDictionaryLearning fit-scheduling atoms adapted from scikit-learn."""

from __future__ import annotations

import math

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dictionary_learning_minibatch_fit_counters,
    witness_dictionary_learning_minibatch_inner_stat_buffers,
    witness_dictionary_learning_minibatch_steps_per_iter,
    witness_dictionary_learning_minibatch_total_steps,
)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _dtype_name_valid(value: object) -> bool:
    return value in {"float32", "float64"}


def _buffers_valid(result: object, n_features: int, n_components: int, dtype_name: str) -> bool:
    if not (isinstance(result, tuple) and len(result) == 2 and _dtype_name_valid(dtype_name)):
        return False
    A, B = result
    dtype = np.dtype(dtype_name)
    A_values = np.asarray(A)
    B_values = np.asarray(B)
    return bool(
        A_values.shape == (int(n_components), int(n_components))
        and B_values.shape == (int(n_features), int(n_components))
        and A_values.dtype == dtype
        and B_values.dtype == dtype
        and np.all(A_values == 0)
        and np.all(B_values == 0)
    )


def _fit_counters_valid(result: object, final_step_index: int, steps_per_iter: int) -> bool:
    if not (isinstance(result, tuple) and len(result) == 2):
        return False
    n_steps, n_iter = result
    expected_steps = int(final_step_index) + 1
    expected_iter = math.ceil(expected_steps / int(steps_per_iter))
    return bool(
        _positive_int(n_steps)
        and isinstance(n_iter, float)
        and np.isfinite(n_iter)
        and float(n_iter) == float(expected_iter)
        and int(n_steps) == expected_steps
    )


@register_atom(witness_dictionary_learning_minibatch_inner_stat_buffers)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.require(lambda dtype_name: _dtype_name_valid(dtype_name), "dtype_name must be 'float32' or 'float64'")
@icontract.ensure(
    lambda result, n_features, n_components, dtype_name: _buffers_valid(result, n_features, n_components, dtype_name),
    "result must contain sklearn-shaped zero inner-stat buffers",
)
def dictionary_learning_minibatch_inner_stat_buffers(
    n_features: int,
    n_components: int,
    dtype_name: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Allocate sklearn's zero-initialized A and B inner-stat buffers."""
    dtype = np.dtype(dtype_name)
    return (
        np.zeros((int(n_components), int(n_components)), dtype=dtype),
        np.zeros((int(n_features), int(n_components)), dtype=dtype),
    )


@register_atom(witness_dictionary_learning_minibatch_steps_per_iter)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda batch_size: _positive_int(batch_size), "batch_size must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "result must be a positive integer")
def dictionary_learning_minibatch_steps_per_iter(
    n_samples: int,
    batch_size: int,
) -> int:
    """Compute sklearn's number of minibatches per outer iteration."""
    return int(math.ceil(int(n_samples) / int(batch_size)))


@register_atom(witness_dictionary_learning_minibatch_total_steps)
@icontract.require(lambda max_iter: _nonnegative_int(max_iter), "max_iter must be a nonnegative integer")
@icontract.require(lambda steps_per_iter: _positive_int(steps_per_iter), "steps_per_iter must be a positive integer")
@icontract.ensure(lambda result: _nonnegative_int(result), "result must be a nonnegative integer")
def dictionary_learning_minibatch_total_steps(
    max_iter: int,
    steps_per_iter: int,
) -> int:
    """Compute sklearn's total minibatch step budget."""
    return int(max_iter) * int(steps_per_iter)


@register_atom(witness_dictionary_learning_minibatch_fit_counters)
@icontract.require(lambda final_step_index: _nonnegative_int(final_step_index), "final_step_index must be a nonnegative integer")
@icontract.require(lambda steps_per_iter: _positive_int(steps_per_iter), "steps_per_iter must be a positive integer")
@icontract.ensure(
    lambda result, final_step_index, steps_per_iter: _fit_counters_valid(result, final_step_index, steps_per_iter),
    "result must match sklearn's n_steps_/n_iter_ derivation",
)
def dictionary_learning_minibatch_fit_counters(
    final_step_index: int,
    steps_per_iter: int,
) -> tuple[int, float]:
    """Derive sklearn's final n_steps_ and n_iter_ counters."""
    n_steps = int(final_step_index) + 1
    n_iter = float(math.ceil(n_steps / int(steps_per_iter)))
    return n_steps, n_iter
