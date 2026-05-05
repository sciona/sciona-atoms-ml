"""Ghost witnesses for partial-dependence column-lookup shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_nonunique_column_guard_required(
    column_index: object,
) -> AbstractArray:
    """Describe the non-unique-dataframe-column guard predicate."""
    del column_index
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_nonunique_column_message(
    columns: tuple[object, ...],
) -> AbstractArray:
    """Describe the non-unique-dataframe-column ValueError message."""
    if len(columns) < 1:
        raise ValueError("columns must be nonempty")
    return AbstractArray(shape=(), dtype="str")


def witness_partial_dependence_column_indices_appended(
    column_indices: tuple[int, ...],
    column_index: int,
) -> AbstractArray:
    """Describe the progressively appended dataframe column indices."""
    del column_index
    return AbstractArray(shape=(len(column_indices) + 1,), dtype="int64")


def witness_partial_dependence_missing_column_message(
    missing_column: object,
) -> AbstractArray:
    """Describe the missing-dataframe-column ValueError message."""
    del missing_column
    return AbstractArray(shape=(), dtype="str")
