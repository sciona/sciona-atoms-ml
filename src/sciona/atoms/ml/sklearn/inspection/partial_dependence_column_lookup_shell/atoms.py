"""Partial-dependence column-lookup shell atoms adapted from scikit-learn."""

from __future__ import annotations

import numbers

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_column_indices_appended,
    witness_partial_dependence_missing_column_message,
    witness_partial_dependence_nonunique_column_guard_required,
    witness_partial_dependence_nonunique_column_message,
)


def _nonempty_tuple(value: object) -> bool:
    return isinstance(value, tuple) and len(value) >= 1


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonnegative_int_tuple(value: object) -> bool:
    return isinstance(value, tuple) and all(_nonnegative_int(item) for item in value)


@register_atom(witness_partial_dependence_nonunique_column_guard_required)
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_nonunique_column_guard_required(
    column_index: object,
) -> bool:
    """Decide whether sklearn raises the non-unique-dataframe-column ValueError."""
    return not isinstance(column_index, numbers.Integral)


@register_atom(witness_partial_dependence_nonunique_column_message)
@icontract.require(lambda columns: _nonempty_tuple(columns), "columns must be a nonempty tuple")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty string")
def partial_dependence_nonunique_column_message(
    columns: tuple[object, ...],
) -> str:
    """Format sklearn's non-unique dataframe-column ValueError message."""
    return f"Selected columns, {list(columns)}, are not unique in dataframe"


@register_atom(witness_partial_dependence_column_indices_appended)
@icontract.require(lambda column_indices: _nonnegative_int_tuple(column_indices), "column_indices must be a tuple of nonnegative integers")
@icontract.require(lambda column_index: _nonnegative_int(column_index), "column_index must be a nonnegative integer")
@icontract.ensure(
    lambda result, column_indices, column_index: isinstance(result, tuple)
    and result == tuple(column_indices) + (int(column_index),),
    "result must append column_index to column_indices",
)
def partial_dependence_column_indices_appended(
    column_indices: tuple[int, ...],
    column_index: int,
) -> tuple[int, ...]:
    """Append one resolved dataframe column index in sklearn's lookup loop."""
    return tuple(column_indices) + (int(column_index),)


@register_atom(witness_partial_dependence_missing_column_message)
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty string")
def partial_dependence_missing_column_message(
    missing_column: object,
) -> str:
    """Return sklearn's missing-dataframe-column ValueError message."""
    del missing_column
    return "A given column is not a column of the dataframe"
