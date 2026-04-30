"""Ghost witnesses for BIRCH insert-routing helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_birch_insert_closest_scores(
    centroids: AbstractArray,
    squared_norm: AbstractArray,
    candidate_centroid: AbstractArray,
) -> AbstractArray:
    """Describe the 1D score vector used to pick the closest BIRCH subcluster."""
    if len(centroids.shape) != 2 or int(centroids.shape[0]) < 1 or int(centroids.shape[1]) < 1:
        raise ValueError("centroids must be a nonempty 2D matrix")
    if len(squared_norm.shape) != 1 or int(squared_norm.shape[0]) != int(centroids.shape[0]):
        raise ValueError("squared_norm must match the centroid count")
    if len(candidate_centroid.shape) != 1 or int(candidate_centroid.shape[0]) != int(centroids.shape[1]):
        raise ValueError("candidate_centroid must match the centroid width")
    return AbstractArray(shape=(int(centroids.shape[0]),), dtype="float64")


def witness_birch_insert_closest_index(closest_scores: AbstractArray) -> AbstractArray:
    """Describe the integer closest-subcluster index selected from BIRCH insert scores."""
    if len(closest_scores.shape) != 1 or int(closest_scores.shape[0]) < 1:
        raise ValueError("closest_scores must be a nonempty 1D vector")
    return AbstractArray(shape=(), dtype="int64")


def witness_birch_insert_child_update_required(has_child: bool, split_child: bool) -> AbstractArray:
    """Describe the Boolean branch for the recursive child-update path in BIRCH insertion."""
    del has_child, split_child
    return AbstractArray(shape=(), dtype="bool")


def witness_birch_insert_child_split_required(has_child: bool, split_child: bool) -> AbstractArray:
    """Describe the Boolean branch for the recursive child-split path in BIRCH insertion."""
    del has_child, split_child
    return AbstractArray(shape=(), dtype="bool")


def witness_birch_insert_append_without_split_required(
    has_child: bool,
    merged: bool,
    current_count: int,
    branching_factor: int,
) -> AbstractArray:
    """Describe the Boolean branch for the append-without-split path in BIRCH insertion."""
    del has_child, merged, current_count, branching_factor
    return AbstractArray(shape=(), dtype="bool")


def witness_birch_insert_append_with_split_required(
    has_child: bool,
    merged: bool,
    current_count: int,
    branching_factor: int,
) -> AbstractArray:
    """Describe the Boolean branch for the append-and-split path in BIRCH insertion."""
    del has_child, merged, current_count, branching_factor
    return AbstractArray(shape=(), dtype="bool")


def witness_birch_insert_parent_split_required(updated_count: int, branching_factor: int) -> AbstractArray:
    """Describe the Boolean branch for parent overflow after a recursive child split in BIRCH insertion."""
    del updated_count, branching_factor
    return AbstractArray(shape=(), dtype="bool")
