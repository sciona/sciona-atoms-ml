"""Ghost witnesses for partial-dependence feature-name preflight shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_feature_key_is_string(
    feature_key: int | str,
) -> AbstractArray:
    """Describe the string-feature-key branch predicate."""
    del feature_key
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_feature_names_required_guard_required(
    key_is_string: bool,
    feature_names_provided: bool,
) -> AbstractArray:
    """Describe the missing-feature-names guard predicate."""
    del key_is_string
    del feature_names_provided
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_feature_names_required_message(
    feature_name: str,
) -> AbstractArray:
    """Describe the missing-feature-names ValueError message."""
    del feature_name
    return AbstractArray(shape=(), dtype="str")


def witness_partial_dependence_feature_name_missing_guard_required(
    feature_name: str,
    feature_names: tuple[str, ...],
) -> AbstractArray:
    """Describe the missing-feature-name guard predicate."""
    del feature_name
    del feature_names
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_feature_name_missing_message(
    feature_name: str,
) -> AbstractArray:
    """Describe the missing-feature-name ValueError message."""
    del feature_name
    return AbstractArray(shape=(), dtype="str")
