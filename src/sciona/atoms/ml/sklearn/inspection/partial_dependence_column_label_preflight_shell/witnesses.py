"""Ghost witnesses for partial-dependence column-label preflight shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_column_key_uses_label_branch(
    key_dtype: str,
) -> AbstractArray:
    """Describe the label-branch predicate for column key types."""
    del key_dtype
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_dataframe_columns_required_guard_required(
    uses_label_branch: bool,
    has_columns: bool,
) -> AbstractArray:
    """Describe the non-dataframe label-selection guard predicate."""
    del uses_label_branch
    del has_columns
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_dataframe_columns_required_message(
    uses_label_branch: bool,
) -> AbstractArray:
    """Describe the non-dataframe label-selection ValueError message."""
    del uses_label_branch
    return AbstractArray(shape=(), dtype="str")


def witness_partial_dependence_string_column_keys(
    column_name: str,
) -> AbstractArray:
    """Describe sklearn's singleton string-column key sequence."""
    del column_name
    return AbstractArray(shape=(1,), dtype="str")
