"""Partial-dependence column-label preflight shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_column_key_uses_label_branch,
    witness_partial_dependence_dataframe_columns_required_guard_required,
    witness_partial_dependence_dataframe_columns_required_message,
    witness_partial_dependence_string_column_keys,
)


def _key_dtype_token(value: object) -> bool:
    return isinstance(value, str) and value in {"bool", "int", "str", "none"}


def _bool_scalar(value: object) -> bool:
    return isinstance(value, bool)


def _column_name(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


@register_atom(witness_partial_dependence_column_key_uses_label_branch)
@icontract.require(lambda key_dtype: _key_dtype_token(key_dtype), "key_dtype must be one of: bool, int, str, none")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_column_key_uses_label_branch(
    key_dtype: str,
) -> bool:
    """Decide whether sklearn's column-index helper enters the label-based branch."""
    return key_dtype not in ("bool", "int")


@register_atom(witness_partial_dependence_dataframe_columns_required_guard_required)
@icontract.require(lambda uses_label_branch: _bool_scalar(uses_label_branch), "uses_label_branch must be boolean")
@icontract.require(lambda has_columns: _bool_scalar(has_columns), "has_columns must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_dataframe_columns_required_guard_required(
    uses_label_branch: bool,
    has_columns: bool,
) -> bool:
    """Decide whether sklearn rejects label-based column selection without dataframe columns."""
    return uses_label_branch and (not has_columns)


@register_atom(witness_partial_dependence_dataframe_columns_required_message)
@icontract.require(lambda uses_label_branch: _bool_scalar(uses_label_branch), "uses_label_branch must be boolean")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty message")
def partial_dependence_dataframe_columns_required_message(
    uses_label_branch: bool,
) -> str:
    """Format sklearn's non-dataframe string-column ValueError message."""
    del uses_label_branch
    return "Specifying the columns using strings is only supported for dataframes."


@register_atom(witness_partial_dependence_string_column_keys)
@icontract.require(lambda column_name: _column_name(column_name), "column_name must be a nonempty string")
@icontract.ensure(lambda result, column_name: isinstance(result, tuple) and result == (column_name,), "result must wrap one column name in a singleton tuple")
def partial_dependence_string_column_keys(
    column_name: str,
) -> tuple[str, ...]:
    """Wrap one string column key the way sklearn builds its label lookup sequence."""
    return (column_name,)
