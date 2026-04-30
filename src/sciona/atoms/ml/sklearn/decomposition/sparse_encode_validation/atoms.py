"""Sparse-encode validation atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sparse_encode_require_matching_features,
    witness_sparse_encode_require_positive_compatible_algorithm,
)

Algorithm = str

_VALID_ALGORITHMS = {"lasso_lars", "lasso_cd", "lars", "omp", "threshold"}


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _algorithm_valid(algorithm: object) -> bool:
    return bool(isinstance(algorithm, str) and algorithm in _VALID_ALGORITHMS)


def _flag_valid(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_sparse_encode_require_matching_features)
@icontract.require(lambda x_feature_count: _positive_int(x_feature_count), "x_feature_count must be a positive integer")
@icontract.require(lambda dictionary_feature_count: _positive_int(dictionary_feature_count), "dictionary_feature_count must be a positive integer")
@icontract.ensure(lambda result, x_feature_count: result == int(x_feature_count), "validated feature count must equal X's feature count")
def sparse_encode_require_matching_features(
    x_feature_count: int,
    *,
    dictionary_feature_count: int,
) -> int:
    """Apply sparse_encode's feature-count compatibility guard before solver dispatch."""
    if int(dictionary_feature_count) != int(x_feature_count):
        raise ValueError(
            "Dictionary and X have different numbers of features:"
            f"dictionary.shape: ({dictionary_feature_count},) X.shape({x_feature_count},)"
        )
    return int(x_feature_count)


@register_atom(witness_sparse_encode_require_positive_compatible_algorithm)
@icontract.require(lambda algorithm: _algorithm_valid(algorithm), "algorithm must be one of sklearn's sparse-encode modes")
@icontract.require(lambda positive: _flag_valid(positive), "positive must be boolean")
@icontract.ensure(lambda result, algorithm: result == algorithm, "validated algorithm must be returned unchanged")
def sparse_encode_require_positive_compatible_algorithm(
    algorithm: Algorithm,
    *,
    positive: bool,
) -> Algorithm:
    """Apply sparse_encode's positivity guard to the selected coding algorithm."""
    if positive and algorithm in {"omp", "lars"}:
        raise ValueError(f"Positive constraint not supported for '{algorithm}' coding method.")
    return algorithm
