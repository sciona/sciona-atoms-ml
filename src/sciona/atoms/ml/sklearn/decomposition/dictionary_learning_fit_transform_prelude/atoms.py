"""DictionaryLearning fit-transform prelude atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dictionary_learning_fit_transform_method,
    witness_dictionary_learning_fit_transform_n_components,
    witness_dictionary_learning_fit_transform_require_positive_coding_supported,
    witness_dictionary_learning_fit_transform_validated_data,
)

Matrix = NDArray[np.float64]


def _fit_algorithm_valid(value: object) -> bool:
    return value in {"lars", "cd"}


def _positive_int_or_none(value: object) -> bool:
    return value is None or (
        isinstance(value, int) and not isinstance(value, bool) and value >= 1
    )


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


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


@register_atom(witness_dictionary_learning_fit_transform_require_positive_coding_supported)
@icontract.require(lambda fit_algorithm: _fit_algorithm_valid(fit_algorithm), "fit_algorithm must be 'lars' or 'cd'")
@icontract.require(lambda positive_code: isinstance(positive_code, bool), "positive_code must be boolean")
@icontract.require(
    lambda fit_algorithm, positive_code: not (positive_code and fit_algorithm == "lars"),
    "positive_code is not supported for lars coding",
)
@icontract.ensure(lambda result: result is True, "supported positive-coding configurations return True")
def dictionary_learning_fit_transform_require_positive_coding_supported(
    fit_algorithm: str,
    positive_code: bool,
) -> bool:
    """Validate DictionaryLearning.fit_transform positive-coding support before _dict_learning."""
    del positive_code
    return True


@register_atom(witness_dictionary_learning_fit_transform_method)
@icontract.require(lambda fit_algorithm: _fit_algorithm_valid(fit_algorithm), "fit_algorithm must be 'lars' or 'cd'")
@icontract.ensure(lambda result: result in {"lasso_lars", "lasso_cd"}, "result must be the lasso-prefixed fit method label")
def dictionary_learning_fit_transform_method(fit_algorithm: str) -> str:
    """Resolve the internal method label passed into _dict_learning."""
    return "lasso_" + fit_algorithm


@register_atom(witness_dictionary_learning_fit_transform_validated_data)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite nonempty 2D matrix")
@icontract.ensure(
    lambda result, X: _finite_matrix(result)
    and np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(X, dtype=np.float64)),
    "validated data must preserve the supplied finite matrix for already-valid dense inputs",
)
def dictionary_learning_fit_transform_validated_data(X: Matrix) -> Matrix:
    """Expose the validated training matrix passed into _dict_learning for dense finite inputs."""
    return np.asarray(X, dtype=np.float64)


@register_atom(witness_dictionary_learning_fit_transform_n_components)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite nonempty 2D matrix")
@icontract.require(lambda n_components: _positive_int_or_none(n_components), "n_components must be None or a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "resolved n_components must be a positive integer")
def dictionary_learning_fit_transform_n_components(X: Matrix, n_components: int | None) -> int:
    """Resolve DictionaryLearning.fit_transform's effective component count."""
    values = np.asarray(X, dtype=np.float64)
    if n_components is None:
        return int(values.shape[1])
    return int(n_components)
