"""Ghost witnesses for partial-dependence categorical-dispatch shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_categorical_array(
    categorical_features: object,
) -> AbstractArray:
    """Describe sklearn's np.asarray(categorical_features) coercion."""
    del categorical_features
    return AbstractArray(shape=(1,), dtype="object")


def witness_partial_dependence_categorical_bool_branch(
    dtype_kind: str,
) -> AbstractArray:
    """Describe the boolean-mask categorical branch predicate."""
    del dtype_kind
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_categorical_index_or_name_branch(
    dtype_kind: str,
) -> AbstractArray:
    """Describe the integer-or-name categorical branch predicate."""
    del dtype_kind
    return AbstractArray(shape=(), dtype="bool")
