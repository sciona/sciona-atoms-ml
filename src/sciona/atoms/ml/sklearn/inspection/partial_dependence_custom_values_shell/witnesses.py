"""Ghost witnesses for partial-dependence custom-values shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_custom_values_mapping(
    custom_values: dict[object, object] | None = None,
) -> AbstractArray:
    """Describe the defaulted custom_values mapping at partial_dependence entry."""
    del custom_values
    return AbstractArray(shape=(), dtype="object")


def witness_partial_dependence_feature_sequence(
    features: int | str | bool | tuple[object, ...] | list[object],
) -> AbstractArray:
    """Describe the scalar-wrap or sequence-preserve feature normalization shell."""
    del features
    return AbstractArray(shape=(), dtype="object")


def witness_partial_dependence_custom_values_subset_mapping(
    features: tuple[object, ...],
    custom_values: dict[object, object],
) -> AbstractArray:
    """Describe the indexed custom-values subset mapping for selected features."""
    if len(features) < 1:
        raise ValueError("features must be nonempty")
    del custom_values
    return AbstractArray(shape=(), dtype="object")
