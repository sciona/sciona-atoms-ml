"""Ghost witnesses for BIRCH split-partition helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_birch_split_distance_matrix(
    centroids: AbstractArray,
    squared_norm: AbstractArray,
) -> AbstractArray:
    """Describe the squared-distance matrix used by sklearn BIRCH node splitting."""
    if len(centroids.shape) != 2 or int(centroids.shape[0]) < 2 or int(centroids.shape[1]) < 1:
        raise ValueError("centroids must be a 2D matrix with at least two rows")
    if len(squared_norm.shape) != 1 or int(squared_norm.shape[0]) != int(centroids.shape[0]):
        raise ValueError("squared_norm must be a 1D vector matching the centroid count")
    n_clusters = int(centroids.shape[0])
    return AbstractArray(shape=(n_clusters, n_clusters), dtype="float64")


def witness_birch_split_farthest_pair(distance_matrix: AbstractArray) -> AbstractArray:
    """Describe the farthest-pair index tuple selected from a BIRCH split distance matrix."""
    if len(distance_matrix.shape) != 2 or int(distance_matrix.shape[0]) < 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square with at least two rows")
    return AbstractArray(shape=(2,), dtype="int64")


def witness_birch_split_assignment_mask(
    distance_matrix: AbstractArray,
    farthest_pair: AbstractArray,
) -> AbstractArray:
    """Describe sklearn's node1-closer Boolean assignment mask for BIRCH node splitting."""
    if len(distance_matrix.shape) != 2 or int(distance_matrix.shape[0]) < 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square with at least two rows")
    if len(farthest_pair.shape) != 1 or int(farthest_pair.shape[0]) != 2:
        raise ValueError("farthest_pair must contain two indices")
    return AbstractArray(shape=(int(distance_matrix.shape[0]),), dtype="bool")


def witness_birch_split_partition_indices(node1_closer: AbstractArray) -> AbstractArray:
    """Describe the two integer index groups induced by the BIRCH split assignment mask."""
    if len(node1_closer.shape) != 1 or int(node1_closer.shape[0]) < 2:
        raise ValueError("node1_closer must be a 1D mask with at least two entries")
    return AbstractArray(shape=(2,), dtype="object")


def witness_birch_split_partition_stats(
    subcluster_states: tuple[object, ...],
    node1_closer: AbstractArray,
) -> AbstractArray:
    """Describe the aggregate subcluster statistics for the two BIRCH split partitions."""
    if not isinstance(subcluster_states, tuple) or len(subcluster_states) < 2:
        raise ValueError("subcluster_states must be a tuple with at least two entries")
    if len(node1_closer.shape) != 1 or int(node1_closer.shape[0]) != len(subcluster_states):
        raise ValueError("node1_closer must match the number of subcluster states")
    return AbstractArray(shape=(2,), dtype="object")
