"""DictionaryLearning postfit-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dictionary_learning_fit_components,
    witness_dictionary_learning_fit_errors,
    witness_dictionary_learning_fit_n_iter,
    witness_dictionary_learning_fit_transform_output,
)

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]


def _finite_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_vector(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _passthrough_matrix_valid(result: object, source: object) -> bool:
    return _finite_matrix(result) and _finite_matrix(source) and np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(source, dtype=np.float64))


def _passthrough_vector_valid(result: object, source: object) -> bool:
    return _finite_vector(result) and _finite_vector(source) and np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(source, dtype=np.float64))


@register_atom(witness_dictionary_learning_fit_components)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite 2D matrix")
@icontract.ensure(lambda result, dictionary: _passthrough_matrix_valid(result, dictionary), "result must expose the fitted dictionary unchanged")
def dictionary_learning_fit_components(
    dictionary: Matrix,
) -> Matrix:
    """Expose DictionaryLearning.components_ from the already-computed dictionary factor."""
    return np.asarray(dictionary, dtype=np.float64)


@register_atom(witness_dictionary_learning_fit_errors)
@icontract.require(lambda errors: _finite_vector(errors), "errors must be a finite 1D vector")
@icontract.ensure(lambda result, errors: _passthrough_vector_valid(result, errors), "result must expose the fitted error curve unchanged")
def dictionary_learning_fit_errors(
    errors: Vector,
) -> Vector:
    """Expose DictionaryLearning.error_ from the already-computed error history."""
    return np.asarray(errors, dtype=np.float64)


@register_atom(witness_dictionary_learning_fit_n_iter)
@icontract.require(lambda n_iter: _positive_int(n_iter), "n_iter must be a positive integer")
@icontract.ensure(lambda result, n_iter: _positive_int(result) and int(result) == int(n_iter), "result must expose the fitted iteration count unchanged")
def dictionary_learning_fit_n_iter(
    n_iter: int,
) -> int:
    """Expose DictionaryLearning.n_iter_ from the already-computed iteration count."""
    return int(n_iter)


@register_atom(witness_dictionary_learning_fit_transform_output)
@icontract.require(lambda code: _finite_matrix(code), "code must be a finite 2D matrix")
@icontract.ensure(lambda result, code: _passthrough_matrix_valid(result, code), "result must expose the fit_transform code matrix unchanged")
def dictionary_learning_fit_transform_output(
    code: Matrix,
) -> Matrix:
    """Expose DictionaryLearning.fit_transform output from the already-computed code matrix."""
    return np.asarray(code, dtype=np.float64)
