"""Ghost witnesses for partial-dependence categorical-preflight shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_categorical_empty_guard_required(
    size: int,
) -> AbstractArray:
    """Describe the empty categorical_features guard predicate."""
    del size
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_categorical_empty_message(
    size: int,
) -> AbstractArray:
    """Describe the empty categorical_features ValueError message."""
    del size
    return AbstractArray(shape=(), dtype="str")


def witness_partial_dependence_categorical_bool_size_guard_required(
    size: int,
    n_features: int,
) -> AbstractArray:
    """Describe the boolean categorical mask length guard predicate."""
    del size
    del n_features
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_categorical_bool_size_message(
    size: int,
    n_features: int,
) -> AbstractArray:
    """Describe the boolean categorical mask length mismatch message."""
    del size
    del n_features
    return AbstractArray(shape=(), dtype="str")


def witness_partial_dependence_categorical_dtype_supported(
    dtype_kind: str,
) -> AbstractArray:
    """Describe the categorical_features dtype-kind support predicate."""
    del dtype_kind
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_categorical_dtype_message(
    dtype_name: str,
) -> AbstractArray:
    """Describe the unsupported categorical_features dtype message."""
    del dtype_name
    return AbstractArray(shape=(), dtype="str")
