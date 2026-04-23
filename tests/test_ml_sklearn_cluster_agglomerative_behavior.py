from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._agglomerative import _hc_cut


def _children() -> np.ndarray:
    return np.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
        dtype=np.int64,
    )


def _data() -> np.ndarray:
    return np.array(
        [
            [-1.0, -1.0],
            [-0.8, -1.1],
            [1.0, 1.0],
            [1.1, 0.9],
            [3.0, 3.1],
        ],
        dtype=np.float64,
    )


def test_agglomerative_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative import (
        agglomerative_descendent_leaves,
        agglomerative_hc_cut,
        agglomerative_root_node,
    )

    assert callable(agglomerative_descendent_leaves)
    assert callable(agglomerative_hc_cut)
    assert callable(agglomerative_root_node)


def test_root_node_and_descendent_leaves_for_manual_tree() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative import (
        agglomerative_descendent_leaves,
        agglomerative_root_node,
    )

    children = _children()
    assert agglomerative_root_node(children, 4) == 6
    assert np.array_equal(agglomerative_descendent_leaves(0, children, 4), np.array([0], dtype=np.int64))
    assert np.array_equal(agglomerative_descendent_leaves(4, children, 4), np.array([0, 1], dtype=np.int64))
    assert np.array_equal(agglomerative_descendent_leaves(5, children, 4), np.array([2, 3], dtype=np.int64))
    assert np.array_equal(agglomerative_descendent_leaves(6, children, 4), np.array([0, 1, 2, 3], dtype=np.int64))


def test_hc_cut_matches_sklearn_private_helper_for_manual_tree() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative import agglomerative_hc_cut

    children = _children()
    for n_clusters in range(1, 5):
        expected = _hc_cut(n_clusters, children, 4)
        actual = agglomerative_hc_cut(n_clusters, children, 4)
        assert np.array_equal(actual, expected)


def test_hc_cut_matches_sklearn_private_helper_for_fitted_tree() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative import (
        agglomerative_descendent_leaves,
        agglomerative_hc_cut,
        agglomerative_root_node,
    )

    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.0, linkage="ward").fit(_data())
    children = model.children_.astype(np.int64, copy=False)
    n_leaves = int(model.n_leaves_)
    root = agglomerative_root_node(children, n_leaves)

    assert sorted(agglomerative_descendent_leaves(root, children, n_leaves).tolist()) == list(range(n_leaves))
    for n_clusters in [1, 2, 3, n_leaves]:
        expected = _hc_cut(n_clusters, children, n_leaves)
        actual = agglomerative_hc_cut(n_clusters, children, n_leaves)
        assert np.array_equal(actual, expected)


def test_single_leaf_hierarchy() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative import (
        agglomerative_descendent_leaves,
        agglomerative_hc_cut,
        agglomerative_root_node,
    )

    children = np.empty((0, 2), dtype=np.int64)
    assert agglomerative_root_node(children, 1) == 0
    assert np.array_equal(agglomerative_descendent_leaves(0, children, 1), np.array([0], dtype=np.int64))
    assert np.array_equal(agglomerative_hc_cut(1, children, 1), np.array([0], dtype=np.int64))


def test_contracts_reject_invalid_agglomerative_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative import (
        agglomerative_descendent_leaves,
        agglomerative_hc_cut,
        agglomerative_root_node,
    )

    children = _children()

    with pytest.raises(ViolationError):
        agglomerative_root_node(children, 0)

    with pytest.raises(ViolationError):
        agglomerative_root_node(np.array([[0, 1, 2]], dtype=np.int64), 2)

    with pytest.raises(ViolationError):
        agglomerative_root_node(np.array([[0, 2]], dtype=np.int64), 2)

    with pytest.raises(ViolationError):
        agglomerative_descendent_leaves(7, children, 4)

    with pytest.raises(ViolationError):
        agglomerative_hc_cut(0, children, 4)

    with pytest.raises(ViolationError):
        agglomerative_hc_cut(5, children, 4)
