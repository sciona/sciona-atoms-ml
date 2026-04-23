"""Dictionary-learning update helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dictionary_learning_active_update,
    witness_dictionary_learning_sufficient_statistics,
)

DictionaryStatistics = tuple[NDArray[np.float64], NDArray[np.float64]]


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and matrix.shape[0] >= 1 and matrix.shape[1] >= 1 and np.all(np.isfinite(matrix)))


def _stats_inputs_valid(Y: NDArray[np.float64], code: NDArray[np.float64]) -> bool:
    return bool(_finite_matrix(Y) and _finite_matrix(code) and np.asarray(Y).shape[0] == np.asarray(code).shape[0])


def _statistics_valid(result: DictionaryStatistics, Y: NDArray[np.float64], code: NDArray[np.float64]) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    A, B = result
    n_samples, n_features = np.asarray(Y).shape
    code_values = np.asarray(code)
    n_components = code_values.shape[1]
    del n_samples
    return bool(
        np.asarray(A).shape == (n_components, n_components)
        and np.asarray(B).shape == (n_features, n_components)
        and np.all(np.isfinite(A))
        and np.all(np.isfinite(B))
        and np.allclose(A, np.asarray(A).T)
    )


def _active_update_inputs_valid(
    dictionary: NDArray[np.float64],
    A: NDArray[np.float64],
    B: NDArray[np.float64],
) -> bool:
    if not (_finite_matrix(dictionary) and _finite_matrix(A) and _finite_matrix(B)):
        return False
    dict_values = np.asarray(dictionary)
    a_values = np.asarray(A, dtype=np.float64)
    b_values = np.asarray(B)
    n_components, n_features = dict_values.shape
    return bool(
        a_values.shape == (n_components, n_components)
        and b_values.shape == (n_features, n_components)
        and np.allclose(a_values, a_values.T)
        and np.all(np.diag(a_values) > 1e-6)
    )


def _active_update_valid(result: NDArray[np.float64], dictionary: NDArray[np.float64], positive: bool) -> bool:
    updated = np.asarray(result)
    return bool(
        updated.shape == np.asarray(dictionary).shape
        and np.all(np.isfinite(updated))
        and np.all(linalg.norm(updated, axis=1) <= 1.0 + 1e-10)
        and (not positive or np.all(updated >= 0.0))
    )


@register_atom(witness_dictionary_learning_sufficient_statistics)
@icontract.require(lambda Y, code: _stats_inputs_valid(Y, code), "Y and code must be finite matrices with matching sample count")
@icontract.ensure(lambda result, Y, code: _statistics_valid(result, Y, code), "sufficient statistics must have dictionary-learning shapes")
def dictionary_learning_sufficient_statistics(
    Y: NDArray[np.float64],
    code: NDArray[np.float64],
) -> DictionaryStatistics:
    """Compute sklearn dictionary-learning sufficient statistics A and B."""
    y_values = np.asarray(Y, dtype=np.float64)
    code_values = np.asarray(code, dtype=np.float64)
    return (
        np.asarray(code_values.T @ code_values, dtype=np.float64),
        np.asarray(y_values.T @ code_values, dtype=np.float64),
    )


@register_atom(witness_dictionary_learning_active_update)
@icontract.require(lambda dictionary, A, B: _active_update_inputs_valid(dictionary, A, B), "dictionary, A, and B must be compatible with active atoms only")
@icontract.ensure(lambda result, dictionary, positive: _active_update_valid(result, dictionary, positive), "updated dictionary rows must be finite and norm-bounded")
def dictionary_learning_active_update(
    dictionary: NDArray[np.float64],
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    *,
    positive: bool = False,
) -> NDArray[np.float64]:
    """Update active dictionary atoms and project each row to unit norm."""
    updated = np.asarray(dictionary, dtype=np.float64).copy()
    a_values = np.asarray(A, dtype=np.float64)
    b_values = np.asarray(B, dtype=np.float64)
    n_components = updated.shape[0]

    for k in range(n_components):
        updated[k] += (b_values[:, k] - a_values[k] @ updated) / a_values[k, k]
        if positive:
            np.clip(updated[k], 0.0, None, out=updated[k])
        updated[k] /= max(float(linalg.norm(updated[k])), 1.0)

    return np.asarray(updated, dtype=np.float64)
