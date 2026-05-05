"""Partial-dependence feature-name preflight shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_feature_key_is_string,
    witness_partial_dependence_feature_name_missing_guard_required,
    witness_partial_dependence_feature_name_missing_message,
    witness_partial_dependence_feature_names_required_guard_required,
    witness_partial_dependence_feature_names_required_message,
)


def _feature_key(value: object) -> bool:
    return isinstance(value, (int, str)) and not isinstance(value, bool)


def _feature_name(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _feature_name_tuple(value: object) -> bool:
    return bool(
        isinstance(value, tuple)
        and len(value) >= 1
        and all(_feature_name(name) for name in value)
    )


def _bool_scalar(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_partial_dependence_feature_key_is_string)
@icontract.require(lambda feature_key: _feature_key(feature_key), "feature_key must be an int or str")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_feature_key_is_string(
    feature_key: int | str,
) -> bool:
    """Decide whether sklearn's feature-index helper enters the string-name branch."""
    return isinstance(feature_key, str)


@register_atom(witness_partial_dependence_feature_names_required_guard_required)
@icontract.require(lambda key_is_string: _bool_scalar(key_is_string), "key_is_string must be boolean")
@icontract.require(lambda feature_names_provided: _bool_scalar(feature_names_provided), "feature_names_provided must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_feature_names_required_guard_required(
    key_is_string: bool,
    feature_names_provided: bool,
) -> bool:
    """Decide whether sklearn rejects a string feature key without feature names."""
    return key_is_string and (not feature_names_provided)


@register_atom(witness_partial_dependence_feature_names_required_message)
@icontract.require(lambda feature_name: _feature_name(feature_name), "feature_name must be a nonempty string")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty message")
def partial_dependence_feature_names_required_message(
    feature_name: str,
) -> str:
    """Format sklearn's missing-feature-names ValueError for string feature keys."""
    return (
        f"Cannot plot partial dependence for feature {feature_name!r} since "
        "the list of feature names was not provided, neither as "
        "column names of a pandas data-frame nor via the feature_names "
        "parameter."
    )


@register_atom(witness_partial_dependence_feature_name_missing_guard_required)
@icontract.require(lambda feature_name: _feature_name(feature_name), "feature_name must be a nonempty string")
@icontract.require(lambda feature_names: _feature_name_tuple(feature_names), "feature_names must be a nonempty tuple of strings")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_feature_name_missing_guard_required(
    feature_name: str,
    feature_names: tuple[str, ...],
) -> bool:
    """Decide whether sklearn rejects a string feature key missing from feature_names."""
    return feature_name not in feature_names


@register_atom(witness_partial_dependence_feature_name_missing_message)
@icontract.require(lambda feature_name: _feature_name(feature_name), "feature_name must be a nonempty string")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty message")
def partial_dependence_feature_name_missing_message(
    feature_name: str,
) -> str:
    """Format sklearn's missing-feature-name ValueError."""
    return f"Feature {feature_name!r} not in feature_names"
