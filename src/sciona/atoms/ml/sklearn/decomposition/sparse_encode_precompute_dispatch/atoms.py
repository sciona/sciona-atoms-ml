"""Sparse-encode precompute-dispatch atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sparse_encode_dispatched_covariance,
    witness_sparse_encode_dispatched_gram,
    witness_sparse_encode_resolved_copy_cov,
)

Matrix = NDArray[np.float64]
Algorithm = str

_VALID_ALGORITHMS = {"lasso_lars", "lasso_cd", "lars", "omp", "threshold"}


def _finite_matrix(values: object) -> bool:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        matrix.ndim == 2
        and matrix.shape[0] >= 1
        and matrix.shape[1] >= 1
        and np.all(np.isfinite(matrix))
    )


def _algorithm_valid(algorithm: object) -> bool:
    return bool(isinstance(algorithm, str) and algorithm in _VALID_ALGORITHMS)


def _dictionary_valid(dictionary: object) -> bool:
    return _finite_matrix(dictionary)


def _xy_dictionary_valid(X: object, dictionary: object) -> bool:
    return bool(_finite_matrix(X) and _dictionary_valid(dictionary) and np.asarray(X).shape[1] == np.asarray(dictionary).shape[1])


def _optional_gram_valid(gram: object, dictionary: object) -> bool:
    if gram is None:
        return True
    if not (_finite_matrix(gram) and _dictionary_valid(dictionary)):
        return False
    n_components = np.asarray(dictionary).shape[0]
    values = np.asarray(gram)
    return bool(values.shape == (n_components, n_components) and np.allclose(values, values.T))


def _optional_covariance_valid(cov: object, X: object, dictionary: object) -> bool:
    if cov is None:
        return True
    if not _xy_dictionary_valid(X, dictionary) or not _finite_matrix(cov):
        return False
    values = np.asarray(cov)
    return bool(values.shape == (np.asarray(dictionary).shape[0], np.asarray(X).shape[0]))


def _gram_result_valid(result: object, gram: object, dictionary: object, algorithm: Algorithm) -> bool:
    if result is None:
        return bool(gram is None and algorithm == "threshold")
    if not _optional_gram_valid(result, dictionary):
        return False
    if gram is None:
        if algorithm == "threshold":
            return False
        return np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(dictionary, dtype=np.float64) @ np.asarray(dictionary, dtype=np.float64).T)
    return np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(gram, dtype=np.float64))


def _cov_result_valid(result: object, cov: object, X: object, dictionary: object, algorithm: Algorithm) -> bool:
    if result is None:
        return bool(cov is None and algorithm == "lasso_cd")
    if not _optional_covariance_valid(result, X, dictionary):
        return False
    if cov is None:
        if algorithm == "lasso_cd":
            return False
        return np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(dictionary, dtype=np.float64) @ np.asarray(X, dtype=np.float64).T)
    return np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(cov, dtype=np.float64))


@register_atom(witness_sparse_encode_dispatched_gram)
@icontract.require(lambda dictionary: _dictionary_valid(dictionary), "dictionary must be a finite nonempty matrix")
@icontract.require(lambda algorithm: _algorithm_valid(algorithm), "algorithm must be one of sklearn's sparse-encode modes")
@icontract.require(lambda gram, dictionary: _optional_gram_valid(gram, dictionary), "gram must be None or a finite symmetric components-by-components matrix")
@icontract.ensure(lambda result, gram, dictionary, algorithm: _gram_result_valid(result, gram, dictionary, algorithm), "dispatched Gram matrix must match sklearn's precompute branch")
def sparse_encode_dispatched_gram(
    gram: Matrix | None,
    dictionary: Matrix,
    algorithm: Algorithm,
) -> Matrix | None:
    """Resolve the Gram matrix used by _sparse_encode before solver dispatch."""
    if gram is None and algorithm != "threshold":
        values = np.asarray(dictionary, dtype=np.float64)
        return np.asarray(values @ values.T, dtype=np.float64)
    if gram is None:
        return None
    return np.asarray(gram, dtype=np.float64)


@register_atom(witness_sparse_encode_dispatched_covariance)
@icontract.require(lambda X, dictionary: _xy_dictionary_valid(X, dictionary), "X and dictionary must be finite with matching feature count")
@icontract.require(lambda algorithm: _algorithm_valid(algorithm), "algorithm must be one of sklearn's sparse-encode modes")
@icontract.require(lambda cov, X, dictionary: _optional_covariance_valid(cov, X, dictionary), "cov must be None or a finite components-by-samples matrix")
@icontract.ensure(lambda result, cov, X, dictionary, algorithm: _cov_result_valid(result, cov, X, dictionary, algorithm), "dispatched covariance must match sklearn's precompute branch")
def sparse_encode_dispatched_covariance(
    cov: Matrix | None,
    X: Matrix,
    dictionary: Matrix,
    algorithm: Algorithm,
) -> Matrix | None:
    """Resolve the covariance matrix used by _sparse_encode before solver dispatch."""
    if cov is None and algorithm != "lasso_cd":
        return np.asarray(np.asarray(dictionary, dtype=np.float64) @ np.asarray(X, dtype=np.float64).T, dtype=np.float64)
    if cov is None:
        return None
    return np.asarray(cov, dtype=np.float64)


@register_atom(witness_sparse_encode_resolved_copy_cov)
@icontract.require(lambda copy_cov: isinstance(copy_cov, bool), "copy_cov must be boolean")
@icontract.require(lambda algorithm: _algorithm_valid(algorithm), "algorithm must be one of sklearn's sparse-encode modes")
@icontract.ensure(lambda result: isinstance(result, bool), "resolved copy_cov must be boolean")
def sparse_encode_resolved_copy_cov(
    copy_cov: bool,
    cov: Matrix | None,
    algorithm: Algorithm,
) -> bool:
    """Resolve copy_cov after sklearn's covariance precompute branch."""
    if cov is None and algorithm != "lasso_cd":
        return False
    return copy_cov
