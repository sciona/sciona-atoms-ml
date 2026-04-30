from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster._birch import _CFNode, _CFSubcluster, _split_node
from sklearn.metrics.pairwise import euclidean_distances

from sciona.atoms.ml.sklearn.cluster.birch_subcluster_math import birch_subcluster_singleton
from sciona.atoms.ml.sklearn.cluster.birch_split_partition import (
    birch_split_assignment_mask,
    birch_split_distance_matrix,
    birch_split_farthest_pair,
    birch_split_partition_indices,
    birch_split_partition_stats,
)


def _make_node(vectors: list[list[float]]) -> _CFNode:
    node = _CFNode(
        threshold=0.5,
        branching_factor=max(2, len(vectors)),
        is_leaf=True,
        n_features=len(vectors[0]),
        dtype=np.float64,
    )
    for vector in vectors:
        node.append_subcluster(_CFSubcluster(linear_sum=np.asarray(vector, dtype=np.float64)))
    return node


def _indices_from_child_vectors(
    original_vectors: list[list[float]],
    child_subclusters: list[_CFSubcluster],
) -> NDArray[np.int64]:
    remaining = list(enumerate(original_vectors))
    chosen: list[int] = []
    for subcluster in child_subclusters:
        centroid = np.asarray(subcluster.centroid_, dtype=np.float64)
        for pos, (idx, vector) in enumerate(remaining):
            if np.allclose(centroid, np.asarray(vector, dtype=np.float64)):
                chosen.append(idx)
                remaining.pop(pos)
                break
        else:
            raise AssertionError("child subcluster centroid not found in original vectors")
    return np.asarray(chosen, dtype=np.int64)


def test_birch_split_partition_atoms_import() -> None:
    assert callable(birch_split_distance_matrix)
    assert callable(birch_split_farthest_pair)
    assert callable(birch_split_assignment_mask)
    assert callable(birch_split_partition_indices)
    assert callable(birch_split_partition_stats)


def test_birch_split_distance_and_farthest_pair_match_sklearn() -> None:
    node = _make_node([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    expected_dist = euclidean_distances(node.centroids_, Y_norm_squared=node.squared_norm_, squared=True)
    observed_dist = birch_split_distance_matrix(node.centroids_, node.squared_norm_)
    assert np.allclose(observed_dist, expected_dist)
    assert birch_split_farthest_pair(observed_dist) == tuple(np.unravel_index(expected_dist.argmax(), expected_dist.shape))


def test_birch_split_assignment_mask_matches_sklearn_formula_and_tie_override() -> None:
    dist = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    farthest = birch_split_farthest_pair(dist)
    observed = birch_split_assignment_mask(dist, farthest)
    assert farthest == (0, 0)
    assert np.array_equal(observed, np.array([True, False, False], dtype=np.bool_))


def test_birch_split_partition_indices_and_stats_match_private_split_node() -> None:
    vectors = [[0.0, 0.0], [5.0, 0.0], [0.0, 4.0], [1.0, 1.0]]
    node = _make_node(vectors)
    dist = birch_split_distance_matrix(node.centroids_, node.squared_norm_)
    farthest = birch_split_farthest_pair(dist)
    mask = birch_split_assignment_mask(dist, farthest)
    left_idx, right_idx = birch_split_partition_indices(mask)

    states = tuple(birch_subcluster_singleton(np.asarray(vector, dtype=np.float64)) for vector in vectors)
    left_state, right_state = birch_split_partition_stats(states, mask)

    new_subcluster1, new_subcluster2 = _split_node(node, threshold=0.5, branching_factor=4)

    expected_left_idx = _indices_from_child_vectors(vectors, new_subcluster1.child_.subclusters_)
    expected_right_idx = _indices_from_child_vectors(vectors, new_subcluster2.child_.subclusters_)
    assert np.array_equal(left_idx, expected_left_idx)
    assert np.array_equal(right_idx, expected_right_idx)

    assert left_state.n_samples == int(new_subcluster1.n_samples_)
    assert np.allclose(left_state.linear_sum, new_subcluster1.linear_sum_)
    assert np.isclose(left_state.squared_sum, new_subcluster1.squared_sum_)
    assert np.allclose(left_state.centroid, new_subcluster1.centroid_)
    assert np.isclose(left_state.sq_norm, new_subcluster1.sq_norm_)

    assert right_state.n_samples == int(new_subcluster2.n_samples_)
    assert np.allclose(right_state.linear_sum, new_subcluster2.linear_sum_)
    assert np.isclose(right_state.squared_sum, new_subcluster2.squared_sum_)
    assert np.allclose(right_state.centroid, new_subcluster2.centroid_)
    assert np.isclose(right_state.sq_norm, new_subcluster2.sq_norm_)


def test_birch_split_partition_contracts_reject_invalid_inputs() -> None:
    with pytest.raises(ViolationError):
        birch_split_distance_matrix(np.ones((1, 2), dtype=np.float64), np.ones(1, dtype=np.float64))

    with pytest.raises(ViolationError):
        birch_split_farthest_pair(np.ones((2, 3), dtype=np.float64))

    with pytest.raises(ViolationError):
        birch_split_partition_indices(np.array([True, True], dtype=np.bool_))

    with pytest.raises(ViolationError):
        birch_split_partition_stats(
            (
                birch_subcluster_singleton(np.array([0.0], dtype=np.float64)),
                birch_subcluster_singleton(np.array([1.0], dtype=np.float64)),
            ),
            np.array([True], dtype=np.bool_),
        )
