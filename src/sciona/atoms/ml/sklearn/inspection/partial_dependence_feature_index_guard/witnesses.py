"""Ghost witnesses for partial-dependence feature-index guard atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_integer_feature_key_type(
    key_type: str,
) -> AbstractArray:
    """Describe the integer-key-type branch predicate in partial_dependence."""
    del key_type
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_negative_feature_guard_required(
    features: AbstractArray,
) -> AbstractArray:
    """Describe the negative-integer-feature guard predicate in partial_dependence."""
    if len(features.shape) != 1 or int(features.shape[0]) < 1:
        raise ValueError("features must be one-dimensional and nonempty")
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_negative_feature_message(
    n_features: int,
) -> AbstractArray:
    """Describe the negative-integer-feature ValueError message."""
    del n_features
    return AbstractArray(shape=(), dtype="str")
