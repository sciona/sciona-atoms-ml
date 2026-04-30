from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster._birch import _CFNode, _CFSubcluster

from sciona.atoms.ml.sklearn.cluster.birch_node_buffers import (
    birch_append_active_count,
    birch_append_centroids,
    birch_append_squared_norms,
    birch_update_split_centroids,
    birch_update_split_squared_norms,
)


def _make_subcluster(vector: list[float]) -> _CFSubcluster:
    return _CFSubcluster(linear_sum=np.asarray(vector, dtype=np.float64))


def _make_leaf_node(vectors: list[list[float]], branching_factor: int = 4) -> _CFNode:
    node = _CFNode(
        threshold=0.5,
        branching_factor=branching_factor,
        is_leaf=True,
        n_features=len(vectors[0]),
        dtype=np.float64,
    )
    for vector in vectors:
        node.append_subcluster(_make_subcluster(vector))
    return node


def test_birch_node_buffer_atoms_import() -> None:
    assert callable(birch_append_active_count)
    assert callable(birch_append_centroids)
    assert callable(birch_append_squared_norms)
    assert callable(birch_update_split_centroids)
    assert callable(birch_update_split_squared_norms)


def test_birch_append_atoms_match_private_append_subcluster() -> None:
    node = _CFNode(
        threshold=0.5,
        branching_factor=4,
        is_leaf=True,
        n_features=2,
        dtype=np.float64,
    )
    prior_centroids = node.init_centroids_[:0, :].copy()
    prior_squared_norms = node.init_sq_norm_[:0].copy()
    candidate = _make_subcluster([1.0, 2.0])

    node.append_subcluster(candidate)

    assert birch_append_active_count(0) == 1
    assert np.allclose(birch_append_centroids(prior_centroids, candidate.centroid_), node.centroids_)
    assert np.allclose(birch_append_squared_norms(prior_squared_norms, candidate.sq_norm_), node.squared_norm_)


def test_birch_update_split_atoms_match_private_update_split_subclusters() -> None:
    node = _make_leaf_node([[0.0, 0.0], [3.0, 0.0]], branching_factor=4)
    prior_centroids = node.centroids_.copy()
    prior_squared_norms = node.squared_norm_.copy()
    original = node.subclusters_[1]
    replacement = _make_subcluster([2.0, 0.0])
    appended = _make_subcluster([5.0, 0.0])

    node.update_split_subclusters(original, replacement, appended)

    assert np.allclose(
        birch_update_split_centroids(prior_centroids, 1, replacement.centroid_, appended.centroid_),
        node.centroids_,
    )
    assert np.allclose(
        birch_update_split_squared_norms(prior_squared_norms, 1, replacement.sq_norm_, appended.sq_norm_),
        node.squared_norm_,
    )


def test_birch_append_and_split_contracts_reject_invalid_inputs() -> None:
    with pytest.raises(ViolationError):
        birch_append_active_count(-1)

    with pytest.raises(ViolationError):
        birch_append_centroids(np.ones((1, 2), dtype=np.float64), np.ones(3, dtype=np.float64))

    with pytest.raises(ViolationError):
        birch_append_squared_norms(np.array([0.0], dtype=np.float64), -1.0)

    with pytest.raises(ViolationError):
        birch_update_split_centroids(np.ones((1, 2), dtype=np.float64), 1, np.ones(2, dtype=np.float64), np.ones(2, dtype=np.float64))

    with pytest.raises(ViolationError):
        birch_update_split_squared_norms(np.array([0.0], dtype=np.float64), 0, 0.0, -1.0)
