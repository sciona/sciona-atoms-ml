"""MiniBatchDictionaryLearning postfit-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dictionary_learning_minibatch_fit_return_self,
    witness_dictionary_learning_minibatch_postfit_components,
    witness_dictionary_learning_minibatch_postfit_n_iter,
    witness_dictionary_learning_minibatch_postfit_n_steps,
)

Matrix = NDArray[np.float64]


def _finite_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _positive_finite_float(value: object) -> bool:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(scalar) and scalar >= 1.0)


def _passthrough_matrix_valid(result: object, source: object) -> bool:
    return _finite_matrix(result) and _finite_matrix(source) and np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(source, dtype=np.float64))


@register_atom(witness_dictionary_learning_minibatch_postfit_components)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite 2D matrix")
@icontract.ensure(lambda result, dictionary: _passthrough_matrix_valid(result, dictionary), "result must expose the fitted dictionary unchanged")
def dictionary_learning_minibatch_postfit_components(
    dictionary: Matrix,
) -> Matrix:
    """Expose MiniBatchDictionaryLearning.components_ from the already-computed dictionary."""
    return np.asarray(dictionary, dtype=np.float64)


@register_atom(witness_dictionary_learning_minibatch_postfit_n_steps)
@icontract.require(lambda n_steps: _positive_int(n_steps), "n_steps must be a positive integer")
@icontract.ensure(lambda result, n_steps: _positive_int(result) and int(result) == int(n_steps), "result must expose n_steps unchanged")
def dictionary_learning_minibatch_postfit_n_steps(
    n_steps: int,
) -> int:
    """Expose MiniBatchDictionaryLearning.n_steps_ unchanged."""
    return int(n_steps)


@register_atom(witness_dictionary_learning_minibatch_postfit_n_iter)
@icontract.require(lambda n_iter: _positive_finite_float(n_iter), "n_iter must be a finite float >= 1")
@icontract.ensure(lambda result, n_iter: _positive_finite_float(result) and float(result) == float(n_iter), "result must expose n_iter unchanged")
def dictionary_learning_minibatch_postfit_n_iter(
    n_iter: float,
) -> float:
    """Expose MiniBatchDictionaryLearning.n_iter_ unchanged."""
    return float(n_iter)


@register_atom(witness_dictionary_learning_minibatch_fit_return_self)
@icontract.require(lambda estimator_token: isinstance(estimator_token, str) and estimator_token != "", "estimator_token must be a nonempty string")
@icontract.ensure(lambda result, estimator_token: isinstance(result, str) and result == estimator_token, "result must return the estimator token unchanged")
def dictionary_learning_minibatch_fit_return_self(
    estimator_token: str,
) -> str:
    """Model the fit method returning self after postfit state mutation."""
    return estimator_token
