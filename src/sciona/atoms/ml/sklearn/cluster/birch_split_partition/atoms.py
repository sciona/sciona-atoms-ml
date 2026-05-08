"""BIRCH split-partition helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from ..birch_subcluster_math import BirchSubclusterStats, birch_subcluster_update
from .witnesses import (
    witness_birch_split_assignment_mask,
    witness_birch_split_distance_matrix,
    witness_birch_split_farthest_pair,
    witness_birch_split_partition_indices,
    witness_birch_split_partition_stats,
)

def _finite_centroid_matrix(value: object) -> bool:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and matrix.shape[0] >= 2 and matrix.shape[1] >= 1 and np.all(np.isfinite(matrix)))

def _squared_norm_vector_valid(value: object, centroids: NDArray[np.float64]) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(vector.ndim == 1 and vector.shape[0] == np.asarray(centroids).shape[0] and np.all(np.isfinite(vector)) and np.all(vector >= 0.0))

def _distance_matrix_valid(value: object) -> bool:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and matrix.shape[0] >= 2 and matrix.shape[0] == matrix.shape[1] and np.all(np.isfinite(matrix)) and np.all(matrix >= 0.0))

def _farthest_pair_valid(value: object, n_clusters: int) -> bool:
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    return bool(all(isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < n_clusters for idx in value))

def _assignment_mask_valid(value: object, n_clusters: int) -> bool:
    mask = np.asarray(value)
    return bool(mask.ndim == 1 and mask.shape[0] == n_clusters and mask.dtype == np.bool_)

def _partition_indices_valid(value: object, n_clusters: int) -> bool:
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    left, right = value
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.ndim != 1 or right_values.ndim != 1:
        return False
    if not np.issubdtype(left_values.dtype, np.integer) or not np.issubdtype(right_values.dtype, np.integer):
        return False
    combined = np.concatenate((left_values, right_values))
    return bool(
        left_values.shape[0] >= 1
        and right_values.shape[0] >= 1
        and combined.shape[0] == n_clusters
        and np.array_equal(np.sort(combined.astype(np.int64)), np.arange(n_clusters, dtype=np.int64))
    )

def _subcluster_state_valid(value: object) -> bool:
    if not isinstance(value, BirchSubclusterStats):
        return False
    return bool(
        isinstance(value.n_samples, int)
        and value.n_samples >= 1
        and np.asarray(value.linear_sum).ndim == 1
        and np.asarray(value.centroid).shape == np.asarray(value.linear_sum).shape
    )

def _subcluster_state_tuple_valid(value: object) -> bool:
    if not isinstance(value, tuple) or len(value) < 2:
        return False
    if not all(_subcluster_state_valid(item) for item in value):
        return False
    widths = {np.asarray(item.linear_sum).shape for item in value}
    return len(widths) == 1

def _partition_stats_valid(value: object, width: int) -> bool:
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    return bool(
        all(_subcluster_state_valid(item) for item in value)
        and all(np.asarray(item.linear_sum).shape == (width,) for item in value)
    )

@register_atom(witness_birch_split_distance_matrix)
@icontract.require(lambda centroids: _finite_centroid_matrix(centroids), "centroids must be a finite 2D matrix with at least two rows")
@icontract.require(lambda squared_norm, centroids: _squared_norm_vector_valid(squared_norm, centroids), "squared_norm must be a finite nonnegative vector matching centroid count")
@icontract.ensure(lambda result: _distance_matrix_valid(result), "result must be a finite nonnegative square distance matrix")
def birch_split_distance_matrix(
    centroids: NDArray[np.float64],
    squared_norm: NDArray[np.float64],
) -> NDArray[np.float64]:
    from sklearn.metrics.pairwise import euclidean_distances
    """Compute sklearn's squared centroid-distance matrix for BIRCH node splitting."""
    return np.asarray(
        euclidean_distances(
            np.asarray(centroids, dtype=np.float64),
            Y_norm_squared=np.asarray(squared_norm, dtype=np.float64),
            squared=True,
        ),
        dtype=np.float64,
    )

@register_atom(witness_birch_split_farthest_pair)
@icontract.require(lambda distance_matrix: _distance_matrix_valid(distance_matrix), "distance_matrix must be a finite nonnegative square matrix")
@icontract.ensure(lambda result, distance_matrix: _farthest_pair_valid(result, np.asarray(distance_matrix).shape[0]), "result must be a valid pair of cluster indices")
def birch_split_farthest_pair(distance_matrix: NDArray[np.float64]) -> tuple[int, int]:
    """Select sklearn's farthest centroid-pair index tuple for BIRCH node splitting."""
    dist_values = np.asarray(distance_matrix, dtype=np.float64)
    n_clusters = int(dist_values.shape[0])
    farthest_idx = np.unravel_index(int(dist_values.argmax()), (n_clusters, n_clusters))
    return int(farthest_idx[0]), int(farthest_idx[1])

@register_atom(witness_birch_split_assignment_mask)
@icontract.require(lambda distance_matrix: _distance_matrix_valid(distance_matrix), "distance_matrix must be a finite nonnegative square matrix")
@icontract.require(lambda distance_matrix, farthest_pair: _farthest_pair_valid(farthest_pair, np.asarray(distance_matrix).shape[0]), "farthest_pair must be valid for the distance matrix")
@icontract.ensure(lambda result, distance_matrix: _assignment_mask_valid(result, np.asarray(distance_matrix).shape[0]), "result must be a Boolean assignment mask")
def birch_split_assignment_mask(
    distance_matrix: NDArray[np.float64],
    farthest_pair: tuple[int, int],
) -> NDArray[np.bool_]:
    """Compute sklearn's node1-closer assignment mask for BIRCH node splitting."""
    dist_values = np.asarray(distance_matrix, dtype=np.float64)
    node1_dist, node2_dist = dist_values[(farthest_pair,)]
    node1_closer = np.asarray(node1_dist < node2_dist, dtype=np.bool_)
    node1_closer[farthest_pair[0]] = True
    return node1_closer

@register_atom(witness_birch_split_partition_indices)
@icontract.require(lambda node1_closer: _assignment_mask_valid(node1_closer, np.asarray(node1_closer).shape[0]), "node1_closer must be a Boolean 1D mask")
@icontract.require(lambda node1_closer: np.any(node1_closer) and np.any(~np.asarray(node1_closer, dtype=np.bool_)), "node1_closer must split indices into two nonempty groups")
@icontract.ensure(lambda result, node1_closer: _partition_indices_valid(result, np.asarray(node1_closer).shape[0]), "result must partition all indices into two nonempty integer groups")
def birch_split_partition_indices(
    node1_closer: NDArray[np.bool_],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Partition centroid indices into sklearn's node1 and node2 split groups."""
    mask = np.asarray(node1_closer, dtype=np.bool_)
    all_indices = np.arange(mask.shape[0], dtype=np.int64)
    return all_indices[mask], all_indices[~mask]

@register_atom(witness_birch_split_partition_stats)
@icontract.require(lambda subcluster_states: _subcluster_state_tuple_valid(subcluster_states), "subcluster_states must be a tuple of compatible BIRCH subcluster summaries")
@icontract.require(lambda subcluster_states, node1_closer: _assignment_mask_valid(node1_closer, len(subcluster_states)), "node1_closer must be a Boolean mask matching subcluster_states")
@icontract.require(lambda node1_closer: np.any(node1_closer) and np.any(~np.asarray(node1_closer, dtype=np.bool_)), "node1_closer must split states into two nonempty groups")
@icontract.ensure(lambda result, subcluster_states: _partition_stats_valid(result, np.asarray(subcluster_states[0].linear_sum).shape[0]), "result must contain two valid aggregate BIRCH subcluster summaries")
def birch_split_partition_stats(
    subcluster_states: tuple[BirchSubclusterStats, ...],
    node1_closer: NDArray[np.bool_],
) -> tuple[BirchSubclusterStats, BirchSubclusterStats]:
    """Aggregate sklearn's two BIRCH split partitions into new subcluster statistics."""
    left_indices, right_indices = birch_split_partition_indices(node1_closer)

    left_state = subcluster_states[int(left_indices[0])]
    for index in left_indices[1:]:
        left_state = birch_subcluster_update(left_state, subcluster_states[int(index)])

    right_state = subcluster_states[int(right_indices[0])]
    for index in right_indices[1:]:
        right_state = birch_subcluster_update(right_state, subcluster_states[int(index)])

    return left_state, right_state
