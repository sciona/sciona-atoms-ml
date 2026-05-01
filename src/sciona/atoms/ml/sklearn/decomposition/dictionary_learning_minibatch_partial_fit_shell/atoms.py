"""MiniBatchDictionaryLearning partial-fit shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dictionary_learning_minibatch_partial_fit_components,
    witness_dictionary_learning_minibatch_partial_fit_existing_dictionary,
    witness_dictionary_learning_minibatch_partial_fit_first_call,
    witness_dictionary_learning_minibatch_partial_fit_initial_dictionary,
    witness_dictionary_learning_minibatch_partial_fit_initial_inner_stats,
    witness_dictionary_learning_minibatch_partial_fit_initial_n_steps,
    witness_dictionary_learning_minibatch_partial_fit_reset_required,
    witness_dictionary_learning_minibatch_partial_fit_updated_n_steps,
)

Matrix = NDArray[np.float64]
StatsPair = tuple[NDArray[np.float64], NDArray[np.float64]]


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _supported_dtype_name(value: object) -> bool:
    return bool(isinstance(value, str) and value in {"float64", "float32"})


def _finite_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _passthrough_matrix_valid(result: object, source: object) -> bool:
    return _finite_matrix(result) and _finite_matrix(source) and np.array_equal(
        np.asarray(result, dtype=np.float64), np.asarray(source, dtype=np.float64)
    )


def _initial_inner_stats_valid(result: object, n_components: int, n_features: int, dtype_name: str) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    A, B = result
    dtype = np.dtype(dtype_name)
    return bool(
        isinstance(A, np.ndarray)
        and isinstance(B, np.ndarray)
        and A.shape == (n_components, n_components)
        and B.shape == (n_features, n_components)
        and A.dtype == dtype
        and B.dtype == dtype
        and np.all(A == 0)
        and np.all(B == 0)
    )


@register_atom(witness_dictionary_learning_minibatch_partial_fit_first_call)
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def dictionary_learning_minibatch_partial_fit_first_call(has_components: bool) -> bool:
    """Resolve whether MiniBatchDictionaryLearning.partial_fit is running on an unfitted instance."""
    return not has_components


@register_atom(witness_dictionary_learning_minibatch_partial_fit_reset_required)
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def dictionary_learning_minibatch_partial_fit_reset_required(has_components: bool) -> bool:
    """Resolve validate_data(reset=...) for MiniBatchDictionaryLearning.partial_fit."""
    return not has_components


@register_atom(witness_dictionary_learning_minibatch_partial_fit_initial_n_steps)
@icontract.require(lambda first_call: first_call is True, "first_call must be True for first-pass n_steps_ initialization")
@icontract.ensure(lambda result: _nonnegative_int(result) and int(result) == 0, "initial n_steps_ must be zero")
def dictionary_learning_minibatch_partial_fit_initial_n_steps(first_call: bool) -> int:
    """Initialize MiniBatchDictionaryLearning.n_steps_ on the first partial_fit call."""
    del first_call
    return 0


@register_atom(witness_dictionary_learning_minibatch_partial_fit_initial_inner_stats)
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(lambda dtype_name: _supported_dtype_name(dtype_name), "dtype_name must be float64 or float32")
@icontract.ensure(
    lambda result, n_components, n_features, dtype_name: _initial_inner_stats_valid(
        result, n_components, n_features, dtype_name
    ),
    "initial inner-stat buffers must be zero arrays with sklearn's shapes and dtype",
)
def dictionary_learning_minibatch_partial_fit_initial_inner_stats(
    n_components: int,
    n_features: int,
    dtype_name: str,
) -> StatsPair:
    """Allocate sklearn's first-call partial-fit inner-stat buffers."""
    dtype = np.dtype(dtype_name)
    return (
        np.zeros((n_components, n_components), dtype=dtype),
        np.zeros((n_features, n_components), dtype=dtype),
    )


@register_atom(witness_dictionary_learning_minibatch_partial_fit_initial_dictionary)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite 2D matrix")
@icontract.ensure(
    lambda result, dictionary: _passthrough_matrix_valid(result, dictionary),
    "result must expose the initialized dictionary unchanged",
)
def dictionary_learning_minibatch_partial_fit_initial_dictionary(dictionary: Matrix) -> Matrix:
    """Expose the initialized dictionary chosen on the first partial_fit call."""
    return np.asarray(dictionary, dtype=np.float64)


@register_atom(witness_dictionary_learning_minibatch_partial_fit_existing_dictionary)
@icontract.require(lambda components: _finite_matrix(components), "components must be a finite 2D matrix")
@icontract.ensure(
    lambda result, components: _passthrough_matrix_valid(result, components),
    "result must expose the existing fitted components unchanged",
)
def dictionary_learning_minibatch_partial_fit_existing_dictionary(components: Matrix) -> Matrix:
    """Expose the fitted dictionary reused on later partial_fit calls."""
    return np.asarray(components, dtype=np.float64)


@register_atom(witness_dictionary_learning_minibatch_partial_fit_components)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite 2D matrix")
@icontract.ensure(
    lambda result, dictionary: _passthrough_matrix_valid(result, dictionary),
    "result must expose the updated dictionary as components_",
)
def dictionary_learning_minibatch_partial_fit_components(dictionary: Matrix) -> Matrix:
    """Expose MiniBatchDictionaryLearning.components_ after one minibatch update."""
    return np.asarray(dictionary, dtype=np.float64)


@register_atom(witness_dictionary_learning_minibatch_partial_fit_updated_n_steps)
@icontract.require(lambda n_steps: _nonnegative_int(n_steps), "n_steps must be a nonnegative integer")
@icontract.ensure(
    lambda result, n_steps: _positive_int(result) and int(result) == int(n_steps) + 1,
    "updated n_steps_ must increment by one after the minibatch step",
)
def dictionary_learning_minibatch_partial_fit_updated_n_steps(n_steps: int) -> int:
    """Increment MiniBatchDictionaryLearning.n_steps_ after one partial-fit minibatch update."""
    return int(n_steps) + 1
