from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import cluster_optics_dbscan as sklearn_cluster_optics_dbscan
from sklearn.cluster import compute_optics_graph


def test_optics_extract_atom_import() -> None:
    from sciona.atoms.ml.sklearn.cluster import cluster_optics_dbscan

    assert callable(cluster_optics_dbscan)


def test_cluster_optics_dbscan_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster import cluster_optics_dbscan

    X = np.array([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]], dtype=np.float64)
    ordering, core_distances, reachability, _ = compute_optics_graph(
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

    result = cluster_optics_dbscan(
        reachability=reachability,
        core_distances=core_distances,
        ordering=ordering,
        eps=4.5,
    )
    expected = sklearn_cluster_optics_dbscan(
        reachability=reachability,
        core_distances=core_distances,
        ordering=ordering,
        eps=4.5,
    )

    assert np.array_equal(result, expected)


def test_cluster_optics_dbscan_marks_far_noncore_as_noise() -> None:
    from sciona.atoms.ml.sklearn.cluster import cluster_optics_dbscan

    reachability = np.array([np.inf, 0.2, 5.0, 0.1], dtype=np.float64)
    core_distances = np.array([0.1, 0.2, 5.0, 0.1], dtype=np.float64)
    ordering = np.array([0, 1, 2, 3], dtype=np.int_)

    labels = cluster_optics_dbscan(
        reachability=reachability,
        core_distances=core_distances,
        ordering=ordering,
        eps=1.0,
    )

    assert np.array_equal(labels, np.array([0, 0, -1, 0]))


def test_cluster_optics_dbscan_rejects_non_permutation_ordering() -> None:
    from sciona.atoms.ml.sklearn.cluster import cluster_optics_dbscan

    with pytest.raises(Exception):
        cluster_optics_dbscan(
            reachability=np.array([np.inf, 0.2], dtype=np.float64),
            core_distances=np.array([0.1, 0.2], dtype=np.float64),
            ordering=np.array([0, 0], dtype=np.int_),
            eps=1.0,
        )
