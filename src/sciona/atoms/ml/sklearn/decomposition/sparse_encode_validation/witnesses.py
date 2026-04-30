"""Ghost witnesses for sparse-encode validation helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_sparse_encode_require_matching_features(
    x_feature_count: int,
    *,
    dictionary_feature_count: int,
) -> AbstractArray:
    """Describe the validated feature count after sparse_encode's shape guard."""
    del dictionary_feature_count
    if x_feature_count < 1:
        raise ValueError("x_feature_count must be positive")
    return AbstractArray(shape=(), dtype="int64", min_val=1.0)


def witness_sparse_encode_require_positive_compatible_algorithm(
    algorithm: str,
    *,
    positive: bool,
) -> AbstractArray:
    """Describe the validated sparse-encode algorithm label after the positivity guard."""
    del positive
    if algorithm not in {"lasso_lars", "lasso_cd", "lars", "omp", "threshold"}:
        raise ValueError("algorithm must be a supported sparse_encode mode")
    return AbstractArray(shape=(), dtype="object")
