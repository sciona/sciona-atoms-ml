"""Partial-dependence feature-name setup shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_default_feature_names,
    witness_partial_dependence_duplicate_feature_names_guard_required,
    witness_partial_dependence_duplicate_feature_names_message,
    witness_partial_dependence_use_column_names_tolist,
    witness_partial_dependence_use_feature_names_tolist,
)


def _bool_scalar(value: object) -> bool:
    return isinstance(value, bool)


def _nonnegative_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _feature_names_tuple(value: object) -> bool:
    return bool(
        isinstance(value, tuple)
        and all(isinstance(name, str) and len(name) >= 1 for name in value)
    )


@register_atom(witness_partial_dependence_use_column_names_tolist)
@icontract.require(lambda has_columns: _bool_scalar(has_columns), "has_columns must be boolean")
@icontract.require(lambda columns_has_tolist: _bool_scalar(columns_has_tolist), "columns_has_tolist must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_use_column_names_tolist(
    has_columns: bool,
    columns_has_tolist: bool,
) -> bool:
    """Decide whether sklearn uses X.columns.tolist() for default feature names."""
    return has_columns and columns_has_tolist


@register_atom(witness_partial_dependence_default_feature_names)
@icontract.require(lambda n_features: _nonnegative_int(n_features), "n_features must be a nonnegative integer")
@icontract.ensure(
    lambda result, n_features: isinstance(result, tuple)
    and result == tuple(f"x{i}" for i in range(int(n_features))),
    "result must be sklearn's default x{i} feature-name tuple",
)
def partial_dependence_default_feature_names(
    n_features: int,
) -> tuple[str, ...]:
    """Build sklearn's default x{i} feature names for array inputs."""
    return tuple(f"x{i}" for i in range(int(n_features)))


@register_atom(witness_partial_dependence_use_feature_names_tolist)
@icontract.require(lambda feature_names_provided: _bool_scalar(feature_names_provided), "feature_names_provided must be boolean")
@icontract.require(lambda feature_names_has_tolist: _bool_scalar(feature_names_has_tolist), "feature_names_has_tolist must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_use_feature_names_tolist(
    feature_names_provided: bool,
    feature_names_has_tolist: bool,
) -> bool:
    """Decide whether sklearn normalizes provided feature_names with tolist()."""
    return feature_names_provided and feature_names_has_tolist


@register_atom(witness_partial_dependence_duplicate_feature_names_guard_required)
@icontract.require(lambda feature_names: _feature_names_tuple(feature_names), "feature_names must be a tuple of nonempty strings")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_duplicate_feature_names_guard_required(
    feature_names: tuple[str, ...],
) -> bool:
    """Decide whether sklearn rejects duplicate feature names."""
    return len(set(feature_names)) != len(feature_names)


@register_atom(witness_partial_dependence_duplicate_feature_names_message)
@icontract.require(lambda feature_names: _feature_names_tuple(feature_names), "feature_names must be a tuple of nonempty strings")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty message")
def partial_dependence_duplicate_feature_names_message(
    feature_names: tuple[str, ...],
) -> str:
    """Format sklearn's duplicate feature_names ValueError message."""
    del feature_names
    return "feature_names should not contain duplicates."
