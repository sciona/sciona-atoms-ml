from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import _agglomerative as sklearn_agglomerative
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph


def test_agglomerative_fit_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_bookkeeping import (
        agglomerative_cluster_count_from_distances,
        agglomerative_labels_from_heads,
        agglomerative_resolve_compute_full_tree,
        agglomerative_resolve_tree_n_clusters,
        agglomerative_return_distance_required,
    )

    assert callable(agglomerative_resolve_compute_full_tree)
    assert callable(agglomerative_resolve_tree_n_clusters)
    assert callable(agglomerative_return_distance_required)
    assert callable(agglomerative_cluster_count_from_distances)
    assert callable(agglomerative_labels_from_heads)


def test_agglomerative_resolve_compute_full_tree_matches_sklearn_auto_rule() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_bookkeeping import agglomerative_resolve_compute_full_tree

    assert agglomerative_resolve_compute_full_tree("auto", True, 5, 1000) is True
    assert agglomerative_resolve_compute_full_tree("auto", True, 500, 1000) is False
    assert agglomerative_resolve_compute_full_tree("auto", False, 500, 1000) is True
    assert agglomerative_resolve_compute_full_tree("auto", True, 500, 1000, distance_threshold=0.8) is True
    assert agglomerative_resolve_compute_full_tree(False, True, 5, 1000) is False


def test_agglomerative_resolve_tree_n_clusters_matches_sklearn_rule() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_bookkeeping import agglomerative_resolve_tree_n_clusters

    assert agglomerative_resolve_tree_n_clusters(7, True) is None
    assert agglomerative_resolve_tree_n_clusters(7, False) == 7


def test_agglomerative_return_distance_required_matches_sklearn_rule() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_bookkeeping import agglomerative_return_distance_required

    assert agglomerative_return_distance_required(distance_threshold=None, compute_distances=False) is False
    assert agglomerative_return_distance_required(distance_threshold=1.5, compute_distances=False) is True
    assert agglomerative_return_distance_required(distance_threshold=None, compute_distances=True) is True


def test_agglomerative_cluster_count_from_distances_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_bookkeeping import agglomerative_cluster_count_from_distances

    X, _ = make_blobs(n_samples=25, centers=3, cluster_std=0.6, random_state=7)
    threshold = 2.2
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage="average",
        compute_distances=True,
    ).fit(X)

    result = agglomerative_cluster_count_from_distances(model.distances_.astype(np.float64), distance_threshold=threshold)
    assert result == model.n_clusters_


def test_agglomerative_labels_from_heads_matches_sklearn_early_stop_relabeling() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_bookkeeping import (
        agglomerative_labels_from_heads,
        agglomerative_resolve_compute_full_tree,
        agglomerative_resolve_tree_n_clusters,
    )

    X, _ = make_blobs(n_samples=18, centers=3, cluster_std=0.5, random_state=11)
    connectivity = kneighbors_graph(X, n_neighbors=4, include_self=False)
    n_samples = X.shape[0]
    compute_full_tree = agglomerative_resolve_compute_full_tree("auto", True, 3, n_samples)
    tree_n_clusters = agglomerative_resolve_tree_n_clusters(3, compute_full_tree)

    children, n_connected_components, n_leaves, parents = sklearn_agglomerative._TREE_BUILDERS["ward"](
        X,
        connectivity=connectivity,
        n_clusters=tree_n_clusters,
        return_distance=False,
    )
    del children, n_connected_components, n_leaves
    heads = sklearn_agglomerative._hierarchical.hc_get_heads(parents, copy=False)
    expected = np.searchsorted(np.unique(np.copy(heads[:n_samples])), np.copy(heads[:n_samples]))

    result = agglomerative_labels_from_heads(np.asarray(heads, dtype=np.int64), n_samples=n_samples)
    assert np.array_equal(result, expected)


def test_contracts_reject_invalid_agglomerative_fit_bookkeeping_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_bookkeeping import (
        agglomerative_cluster_count_from_distances,
        agglomerative_labels_from_heads,
        agglomerative_resolve_compute_full_tree,
        agglomerative_resolve_tree_n_clusters,
    )

    with pytest.raises(ViolationError):
        agglomerative_resolve_compute_full_tree("maybe", True, 4, 10)

    with pytest.raises(ViolationError):
        agglomerative_resolve_tree_n_clusters(0, True)

    with pytest.raises(ViolationError):
        agglomerative_cluster_count_from_distances(np.array([[1.0, 2.0]], dtype=np.float64), distance_threshold=1.0)

    with pytest.raises(ViolationError):
        agglomerative_labels_from_heads(np.array([0, 1], dtype=np.int64), n_samples=3)
