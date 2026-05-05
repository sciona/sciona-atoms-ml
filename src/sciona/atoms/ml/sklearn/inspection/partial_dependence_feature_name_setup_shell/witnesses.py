"""Ghost witnesses for partial-dependence feature-name setup shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_use_column_names_tolist(
    has_columns: bool,
    columns_has_tolist: bool,
) -> AbstractArray:
    """Describe the dataframe-column default-name branch predicate."""
    del has_columns
    del columns_has_tolist
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_default_feature_names(
    n_features: int,
) -> AbstractArray:
    """Describe sklearn's default x{i} feature-name sequence."""
    return AbstractArray(shape=(int(n_features),), dtype="str")


def witness_partial_dependence_use_feature_names_tolist(
    feature_names_provided: bool,
    feature_names_has_tolist: bool,
) -> AbstractArray:
    """Describe the provided-feature-names tolist normalization predicate."""
    del feature_names_provided
    del feature_names_has_tolist
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_duplicate_feature_names_guard_required(
    feature_names: tuple[str, ...],
) -> AbstractArray:
    """Describe the duplicate feature_names guard predicate."""
    del feature_names
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_duplicate_feature_names_message(
    feature_names: tuple[str, ...],
) -> AbstractArray:
    """Describe the duplicate feature_names ValueError message."""
    del feature_names
    return AbstractArray(shape=(), dtype="str")
