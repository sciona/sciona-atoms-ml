from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster._birch import _CFNode, _CFSubcluster

from sciona.atoms.ml.sklearn.cluster.birch_insert_routing import (
    birch_insert_append_with_split_required,
    birch_insert_append_without_split_required,
    birch_insert_child_split_required,
    birch_insert_child_update_required,
    birch_insert_closest_index,
    birch_insert_closest_scores,
    birch_insert_parent_split_required,
)


def _make_leaf_node(vectors: list[list[float]], branching_factor: int) -> _CFNode:
    node = _CFNode(
        threshold=0.5,
        branching_factor=branching_factor,
        is_leaf=True,
        n_features=len(vectors[0]),
        dtype=np.float64,
    )
    for vector in vectors:
        node.append_subcluster(_CFSubcluster(linear_sum=np.asarray(vector, dtype=np.float64)))
    return node


def test_birch_insert_routing_atoms_import() -> None:
    assert callable(birch_insert_closest_scores)
    assert callable(birch_insert_closest_index)
    assert callable(birch_insert_child_update_required)
    assert callable(birch_insert_child_split_required)
    assert callable(birch_insert_append_without_split_required)
    assert callable(birch_insert_append_with_split_required)
    assert callable(birch_insert_parent_split_required)


def test_birch_insert_closest_scores_and_index_match_private_formula() -> None:
    node = _make_leaf_node([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]], branching_factor=4)
    candidate = np.array([1.0, 0.5], dtype=np.float64)
    expected_scores = np.dot(node.centroids_, candidate)
    expected_scores *= -2.0
    expected_scores += node.squared_norm_
    observed_scores = birch_insert_closest_scores(node.centroids_, node.squared_norm_, candidate)
    assert np.allclose(observed_scores, expected_scores)
    assert birch_insert_closest_index(observed_scores) == int(np.argmin(expected_scores))


def test_birch_insert_child_branch_predicates_match_recursive_flags() -> None:
    assert birch_insert_child_update_required(True, False) is True
    assert birch_insert_child_update_required(True, True) is False
    assert birch_insert_child_update_required(False, False) is False

    assert birch_insert_child_split_required(True, True) is True
    assert birch_insert_child_split_required(True, False) is False
    assert birch_insert_child_split_required(False, True) is False


def test_birch_insert_append_branch_predicates_match_leaf_cases() -> None:
    node = _make_leaf_node([[0.0, 0.0]], branching_factor=3)
    merged_candidate = _CFSubcluster(linear_sum=np.array([0.1, 0.0], dtype=np.float64))
    merged = node.subclusters_[0].merge_subcluster(merged_candidate, node.threshold)
    assert merged is True
    assert birch_insert_append_without_split_required(False, merged, 1, 3) is False
    assert birch_insert_append_with_split_required(False, merged, 1, 3) is False

    append_node = _make_leaf_node([[0.0, 0.0]], branching_factor=3)
    far_candidate = _CFSubcluster(linear_sum=np.array([3.0, 0.0], dtype=np.float64))
    split_signal = append_node.insert_cf_subcluster(far_candidate)
    assert split_signal is False
    assert len(append_node.subclusters_) == 2
    assert birch_insert_append_without_split_required(False, False, 1, 3) is True
    assert birch_insert_append_with_split_required(False, False, 1, 3) is False

    overflow_node = _make_leaf_node([[0.0, 0.0], [4.0, 0.0]], branching_factor=2)
    overflow_signal = overflow_node.insert_cf_subcluster(_CFSubcluster(linear_sum=np.array([0.0, 5.0], dtype=np.float64)))
    assert overflow_signal is True
    assert len(overflow_node.subclusters_) == 3
    assert birch_insert_append_without_split_required(False, False, 2, 2) is False
    assert birch_insert_append_with_split_required(False, False, 2, 2) is True


def test_birch_insert_parent_split_required_matches_overflow_guard() -> None:
    assert birch_insert_parent_split_required(4, 3) is True
    assert birch_insert_parent_split_required(3, 3) is False


def test_birch_insert_routing_contracts_reject_invalid_inputs() -> None:
    with pytest.raises(ViolationError):
        birch_insert_closest_scores(np.ones((0, 2), dtype=np.float64), np.ones(0, dtype=np.float64), np.ones(2, dtype=np.float64))

    with pytest.raises(ViolationError):
        birch_insert_closest_index(np.array([], dtype=np.float64))

    with pytest.raises(ViolationError):
        birch_insert_append_without_split_required(False, False, 0, 2)

    with pytest.raises(ViolationError):
        birch_insert_parent_split_required(1, 1)
