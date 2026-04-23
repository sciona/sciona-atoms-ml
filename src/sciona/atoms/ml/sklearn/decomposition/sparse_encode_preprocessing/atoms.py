"""Sparse-encode preprocessing atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sparse_encode_covariance,
    witness_sparse_encode_gram,
    witness_sparse_encode_regularization,
    witness_sparse_encode_threshold,
)

Matrix = NDArray[np.float64]
Algorithm = str

_VALID_ALGORITHMS = {"lasso_lars", "lasso_cd", "lars", "omp", "threshold"}


def _finite_matrix(values: Matrix) -> bool:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and matrix.shape[0] >= 1 and matrix.shape[1] >= 1 and np.all(np.isfinite(matrix)))


def _algorithm_valid(algorithm: Algorithm) -> bool:
    return bool(isinstance(algorithm, str) and algorithm in _VALID_ALGORITHMS)


def _optional_positive_int(value: int | None) -> bool:
    return bool(value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 1))


def _optional_nonnegative_float(value: float | None) -> bool:
    return bool(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0))


def _dictionary_valid(dictionary: Matrix) -> bool:
    return _finite_matrix(dictionary)


def _xy_dictionary_valid(X: Matrix, dictionary: Matrix) -> bool:
    return bool(_finite_matrix(X) and _dictionary_valid(dictionary) and np.asarray(X).shape[1] == np.asarray(dictionary).shape[1])


def _covariance_valid(result: Matrix, X: Matrix, dictionary: Matrix) -> bool:
    return bool(
        np.asarray(result).shape == (np.asarray(dictionary).shape[0], np.asarray(X).shape[0])
        and np.all(np.isfinite(result))
    )


def _gram_valid(result: Matrix, dictionary: Matrix) -> bool:
    values = np.asarray(dictionary)
    gram = np.asarray(result)
    n_components = values.shape[0]
    return bool(
        gram.shape == (n_components, n_components)
        and np.all(np.isfinite(gram))
        and np.allclose(gram, gram.T)
    )


def _regularization_valid(result: float, algorithm: Algorithm, n_components: int) -> bool:
    if algorithm in {"lars", "omp"}:
        return bool(np.isfinite(float(result)) and float(result) >= 1.0 and float(result) <= float(n_components))
    return bool(np.isfinite(float(result)) and float(result) >= 0.0)


def _cov_optional_valid(cov: Matrix | None, X: Matrix, dictionary: Matrix) -> bool:
    return bool(cov is None or _covariance_valid(cov, X, dictionary))


def _threshold_result_valid(result: Matrix, X: Matrix, dictionary: Matrix, positive: bool) -> bool:
    code = np.asarray(result)
    return bool(
        code.shape == (np.asarray(X).shape[0], np.asarray(dictionary).shape[0])
        and np.all(np.isfinite(code))
        and (not positive or np.all(code >= 0.0))
    )


@register_atom(witness_sparse_encode_regularization)
@icontract.require(lambda algorithm: _algorithm_valid(algorithm), "algorithm must be one of sklearn's sparse-encode modes")
@icontract.require(lambda n_features: isinstance(n_features, int) and not isinstance(n_features, bool) and n_features >= 1, "n_features must be a positive integer")
@icontract.require(lambda n_components: isinstance(n_components, int) and not isinstance(n_components, bool) and n_components >= 1, "n_components must be a positive integer")
@icontract.require(lambda n_nonzero_coefs: _optional_positive_int(n_nonzero_coefs), "n_nonzero_coefs must be positive when provided")
@icontract.require(lambda alpha: _optional_nonnegative_float(alpha), "alpha must be nonnegative when provided")
@icontract.ensure(lambda result, algorithm, n_components: _regularization_valid(result, algorithm, n_components), "regularization must be finite and compatible with the algorithm")
def sparse_encode_regularization(
    algorithm: Algorithm,
    *,
    n_features: int,
    n_components: int,
    n_nonzero_coefs: int | None = None,
    alpha: float | None = None,
) -> float:
    """Select sklearn's sparse-encode regularization scalar before solver dispatch."""
    if algorithm in {"lars", "omp"}:
        regularization = n_nonzero_coefs
        if regularization is None:
            regularization = min(max(n_features / 10.0, 1.0), float(n_components))
        return float(regularization)

    regularization = alpha
    if regularization is None:
        regularization = 1.0
    return float(regularization)


@register_atom(witness_sparse_encode_gram)
@icontract.require(lambda dictionary: _dictionary_valid(dictionary), "dictionary must be a finite nonempty matrix")
@icontract.ensure(lambda result, dictionary: _gram_valid(result, dictionary), "Gram matrix must be finite, square, and symmetric")
def sparse_encode_gram(dictionary: Matrix) -> Matrix:
    """Compute the sparse-encode Gram matrix dictionary @ dictionary.T."""
    values = np.asarray(dictionary, dtype=np.float64)
    return np.asarray(values @ values.T, dtype=np.float64)


@register_atom(witness_sparse_encode_covariance)
@icontract.require(lambda X, dictionary: _xy_dictionary_valid(X, dictionary), "X and dictionary must be finite with matching feature count")
@icontract.ensure(lambda result, X, dictionary: _covariance_valid(result, X, dictionary), "covariance must be finite with components-by-samples shape")
def sparse_encode_covariance(X: Matrix, dictionary: Matrix) -> Matrix:
    """Compute the sparse-encode covariance matrix dictionary @ X.T."""
    x_values = np.asarray(X, dtype=np.float64)
    dict_values = np.asarray(dictionary, dtype=np.float64)
    return np.asarray(dict_values @ x_values.T, dtype=np.float64)


@register_atom(witness_sparse_encode_threshold)
@icontract.require(lambda X, dictionary: _xy_dictionary_valid(X, dictionary), "X and dictionary must be finite with matching feature count")
@icontract.require(lambda cov, X, dictionary: _cov_optional_valid(cov, X, dictionary), "cov must match components-by-samples shape when provided")
@icontract.require(lambda alpha: _optional_nonnegative_float(alpha), "alpha must be nonnegative when provided")
@icontract.ensure(lambda result, X, dictionary, positive: _threshold_result_valid(result, X, dictionary, positive), "threshold code must be finite and have sparse-encode shape")
def sparse_encode_threshold(
    X: Matrix,
    dictionary: Matrix,
    *,
    cov: Matrix | None = None,
    alpha: float | None = None,
    positive: bool = False,
) -> Matrix:
    """Apply sklearn's threshold sparse-encode branch from precomputed covariance."""
    x_values = np.asarray(X, dtype=np.float64)
    dict_values = np.asarray(dictionary, dtype=np.float64)
    regularization = sparse_encode_regularization(
        "threshold",
        n_features=int(x_values.shape[1]),
        n_components=int(dict_values.shape[0]),
        alpha=alpha,
    )
    cov_values = sparse_encode_covariance(x_values, dict_values) if cov is None else np.asarray(cov, dtype=np.float64)
    code = (np.sign(cov_values) * np.maximum(np.abs(cov_values) - regularization, 0.0)).T
    if positive:
        np.clip(code, 0.0, None, out=code)
    return np.asarray(code, dtype=np.float64)
