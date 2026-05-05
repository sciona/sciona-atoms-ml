"""Ghost witnesses for partial-dependence feature-index output shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_feature_indices_array(
    column_indices: tuple[int, ...],
) -> AbstractArray:
    """Describe sklearn's C-order feature-index array shell."""
    return AbstractArray(shape=(len(column_indices),), dtype="int64")


def witness_partial_dependence_feature_indices_vector(
    feature_indices_array: AbstractArray,
) -> AbstractArray:
    """Describe sklearn's flattened feature-index vector shell."""
    if int(feature_indices_array.size) < 1:
        raise ValueError("feature_indices_array must be nonempty")
    return AbstractArray(shape=(int(feature_indices_array.size),), dtype="int64")


def witness_partial_dependence_selected_feature_count(
    feature_indices: AbstractArray,
) -> AbstractArray:
    """Describe the selected-feature count after vectorization."""
    if int(feature_indices.size) < 1:
        raise ValueError("feature_indices must be nonempty")
    return AbstractArray(shape=(), dtype="int64", min_val=1.0)
