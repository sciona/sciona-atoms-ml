from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import cluster_optics_xi as sklearn_cluster_optics_xi
from sklearn.cluster import compute_optics_graph


def test_optics_xi_atom_import() -> None:
    from sciona.atoms.ml.sklearn.cluster import cluster_optics_xi

    assert callable(cluster_optics_xi)


def test_cluster_optics_xi_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster import cluster_optics_xi

    X = np.array([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]], dtype=np.float64)
    ordering, _, reachability, predecessor = compute_optics_graph(
        X,
        min_samples=2,
        max_eps=np.inf,
        metric="minkowski",
        p=2,
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        n_jobs=None,
    )

    labels, clusters = cluster_optics_xi(
        reachability=reachability,
        predecessor=predecessor,
        ordering=ordering,
        min_samples=2,
    )
    expected_labels, expected_clusters = sklearn_cluster_optics_xi(
        reachability=reachability,
        predecessor=predecessor,
        ordering=ordering,
        min_samples=2,
    )

    assert np.array_equal(labels, expected_labels)
    assert np.array_equal(clusters, expected_clusters)


def test_cluster_optics_xi_matches_sklearn_fractional_sizes() -> None:
    from sciona.atoms.ml.sklearn.cluster import cluster_optics_xi

    reachability = np.array([np.inf, 4.0, 2.0, 1.0, 2.0, 4.0, np.inf], dtype=np.float64)
    predecessor = np.array([-1, 0, 1, 2, 3, 4, 5], dtype=np.int_)
    ordering = np.arange(7, dtype=np.int_)

    labels, clusters = cluster_optics_xi(
        reachability=reachability,
        predecessor=predecessor,
        ordering=ordering,
        min_samples=0.3,
        min_cluster_size=0.3,
        xi=0.05,
        predecessor_correction=False,
    )
    expected_labels, expected_clusters = sklearn_cluster_optics_xi(
        reachability=reachability,
        predecessor=predecessor,
        ordering=ordering,
        min_samples=0.3,
        min_cluster_size=0.3,
        xi=0.05,
        predecessor_correction=False,
    )

    assert np.array_equal(labels, expected_labels)
    assert np.array_equal(clusters, expected_clusters)


def test_cluster_optics_xi_rejects_bad_ordering() -> None:
    from sciona.atoms.ml.sklearn.cluster import cluster_optics_xi

    with pytest.raises(Exception):
        cluster_optics_xi(
            reachability=np.array([np.inf, 0.2], dtype=np.float64),
            predecessor=np.array([-1, 0], dtype=np.int_),
            ordering=np.array([0, 0], dtype=np.int_),
            min_samples=2,
        )
