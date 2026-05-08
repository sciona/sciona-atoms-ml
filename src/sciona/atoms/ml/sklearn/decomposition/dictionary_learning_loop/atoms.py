"""Dictionary-learning loop helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dictionary_learning_callback_due,
    witness_dictionary_learning_converged,
    witness_dictionary_learning_cost,
    witness_dictionary_learning_resize_factors,
    witness_dictionary_learning_svd_initialize,
)

FactorPair = tuple[NDArray[np.float64], NDArray[np.float64]]

def _finite_matrix(values: object) -> bool:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and matrix.shape[0] >= 1 and matrix.shape[1] >= 1 and np.all(np.isfinite(matrix)))

def _compatible_factors(code: object, dictionary: object) -> bool:
    return bool(_finite_matrix(code) and _finite_matrix(dictionary) and np.asarray(code).shape[1] == np.asarray(dictionary).shape[0])

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)

def _finite_scalar(value: float | int) -> bool:
    return bool(not isinstance(value, bool) and np.isscalar(value) and np.isfinite(float(value)))

def _nonnegative_finite_scalar(value: float | int) -> bool:
    return bool(_finite_scalar(value) and float(value) >= 0.0)

def _svd_result_valid(result: FactorPair, X: NDArray[np.float64]) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    code, dictionary = result
    values = np.asarray(X, dtype=np.float64)
    rank = min(values.shape)
    return bool(
        _finite_matrix(code)
        and _finite_matrix(dictionary)
        and np.asarray(code).shape == (values.shape[0], rank)
        and np.asarray(dictionary).shape == (rank, values.shape[1])
    )

def _resize_result_valid(result: FactorPair, code: NDArray[np.float64], dictionary: NDArray[np.float64], n_components: int) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    next_code, next_dictionary = result
    code_values = np.asarray(code, dtype=np.float64)
    dictionary_values = np.asarray(dictionary, dtype=np.float64)
    return bool(
        _finite_matrix(next_code)
        and _finite_matrix(next_dictionary)
        and np.asarray(next_code).shape == (code_values.shape[0], int(n_components))
        and np.asarray(next_dictionary).shape == (int(n_components), dictionary_values.shape[1])
    )

@register_atom(witness_dictionary_learning_svd_initialize)
@icontract.require(lambda X: _finite_matrix(X), "X must be a nonempty finite matrix")
@icontract.ensure(lambda result, X: _svd_result_valid(result, X), "SVD initialization must return finite code and dictionary factors with rank-compatible shapes")
def dictionary_learning_svd_initialize(X: NDArray[np.float64]) -> FactorPair:
    from sklearn.utils.extmath import svd_flip
    """Initialize dictionary-learning factors from a deterministic SVD."""
    values = np.asarray(X, dtype=np.float64)
    code, singular_values, dictionary = linalg.svd(values, full_matrices=False)
    code, dictionary = svd_flip(code, dictionary)
    scaled_dictionary = singular_values[:, np.newaxis] * dictionary
    return np.asarray(code, dtype=np.float64), np.asarray(scaled_dictionary, dtype=np.float64)

@register_atom(witness_dictionary_learning_resize_factors)
@icontract.require(lambda code, dictionary: _compatible_factors(code, dictionary), "code and dictionary must be finite compatible factors")
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.ensure(lambda result, code, dictionary, n_components: _resize_result_valid(result, code, dictionary, n_components), "resized factors must have the requested component count")
def dictionary_learning_resize_factors(
    code: NDArray[np.float64],
    dictionary: NDArray[np.float64],
    *,
    n_components: int,
) -> FactorPair:
    """Crop or zero-pad initial dictionary-learning factors to n_components."""
    code_values = np.asarray(code, dtype=np.float64)
    dictionary_values = np.asarray(dictionary, dtype=np.float64)
    rank = dictionary_values.shape[0]
    if n_components <= rank:
        return (
            np.asarray(code_values[:, :n_components], dtype=np.float64),
            np.asarray(dictionary_values[:n_components, :], dtype=np.float64),
        )
    padded_code = np.c_[code_values, np.zeros((code_values.shape[0], n_components - rank), dtype=np.float64)]
    padded_dictionary = np.r_[dictionary_values, np.zeros((n_components - rank, dictionary_values.shape[1]), dtype=np.float64)]
    return np.asarray(padded_code, dtype=np.float64), np.asarray(padded_dictionary, dtype=np.float64)

@register_atom(witness_dictionary_learning_cost)
@icontract.require(lambda X: _finite_matrix(X), "X must be a nonempty finite matrix")
@icontract.require(lambda code, dictionary: _compatible_factors(code, dictionary), "code and dictionary must be finite compatible factors")
@icontract.require(lambda X, code, dictionary: np.asarray(X).shape[0] == np.asarray(code).shape[0] and np.asarray(X).shape[1] == np.asarray(dictionary).shape[1], "X must align with code and dictionary")
@icontract.require(lambda alpha: _nonnegative_finite_scalar(alpha), "alpha must be finite and nonnegative")
@icontract.ensure(lambda result: _nonnegative_finite_scalar(result), "cost must be finite and nonnegative")
def dictionary_learning_cost(
    X: NDArray[np.float64],
    code: NDArray[np.float64],
    dictionary: NDArray[np.float64],
    *,
    alpha: float,
) -> float:
    """Compute sklearn's dictionary-learning objective for supplied factors."""
    residual = np.asarray(X, dtype=np.float64) - np.asarray(code, dtype=np.float64) @ np.asarray(dictionary, dtype=np.float64)
    return float(0.5 * np.sum(residual**2) + float(alpha) * np.sum(np.abs(np.asarray(code, dtype=np.float64))))

@register_atom(witness_dictionary_learning_converged)
@icontract.require(lambda previous_cost: _finite_scalar(previous_cost), "previous_cost must be finite")
@icontract.require(lambda current_cost: _finite_scalar(current_cost), "current_cost must be finite")
@icontract.require(lambda current_cost: float(current_cost) >= 0.0, "current_cost must be nonnegative")
@icontract.require(lambda tol: _nonnegative_finite_scalar(tol), "tol must be finite and nonnegative")
@icontract.ensure(lambda result: isinstance(result, bool), "convergence flag must be boolean")
def dictionary_learning_converged(previous_cost: float, current_cost: float, *, tol: float) -> bool:
    """Apply sklearn's cost-delta stopping rule for dictionary learning."""
    delta = float(previous_cost) - float(current_cost)
    return bool(delta < float(tol) * float(current_cost))

@register_atom(witness_dictionary_learning_callback_due)
@icontract.require(lambda iteration: _nonnegative_int(iteration), "iteration must be a nonnegative integer")
@icontract.ensure(lambda result: isinstance(result, bool), "callback flag must be boolean")
def dictionary_learning_callback_due(iteration: int) -> bool:
    """Check whether sklearn invokes the dictionary-learning callback at this iteration."""
    return bool(int(iteration) % 5 == 0)
