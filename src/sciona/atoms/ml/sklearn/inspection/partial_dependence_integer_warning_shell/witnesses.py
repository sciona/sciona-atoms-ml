"""Ghost witnesses for partial-dependence integer-warning shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_integer_warning_required(
    is_categorical: bool,
    dtype_kind: str,
) -> AbstractArray:
    """Describe the one-feature integer-warning predicate in partial_dependence."""
    del is_categorical
    del dtype_kind
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_integer_warning_message(
    feature: int | str | bool,
) -> AbstractArray:
    """Describe the integer-data warning message emitted by partial_dependence."""
    del feature
    return AbstractArray(shape=(), dtype="str")


def witness_partial_dependence_first_integer_warning_feature(
    features: tuple[int | str | bool, ...],
    is_categorical: tuple[bool, ...],
    dtype_kinds: tuple[str, ...],
) -> AbstractArray:
    """Describe the first warning-triggering feature selected by the warning loop."""
    if not (len(features) == len(is_categorical) == len(dtype_kinds)):
        raise ValueError("inputs must have the same length")
    return AbstractArray(shape=(), dtype="object")
