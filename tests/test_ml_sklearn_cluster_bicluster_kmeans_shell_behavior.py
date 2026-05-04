from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster import KMeans, MiniBatchKMeans


def _sample_data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.1],
            [0.2, -0.1],
            [4.0, 4.2],
            [3.8, 4.1],
            [0.1, 0.0],
            [4.2, 3.9],
        ],
        dtype=np.float64,
    )


def test_bicluster_kmeans_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_kmeans_shell import (
        bicluster_kmeans_centroids,
        bicluster_kmeans_labels,
        bicluster_project_cluster_labels,
        bicluster_use_minibatch_kmeans,
    )

    assert callable(bicluster_use_minibatch_kmeans)
    assert callable(bicluster_kmeans_centroids)
    assert callable(bicluster_kmeans_labels)
    assert callable(bicluster_project_cluster_labels)


def test_bicluster_kmeans_shell_matches_fitted_kmeans_models() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_kmeans_shell import (
        bicluster_kmeans_centroids,
        bicluster_kmeans_labels,
        bicluster_project_cluster_labels,
        bicluster_use_minibatch_kmeans,
    )

    data = _sample_data()
    kmeans = KMeans(n_clusters=2, n_init=5, random_state=0).fit(data)
    minibatch = MiniBatchKMeans(n_clusters=2, n_init=5, random_state=0, batch_size=3).fit(data)

    assert bicluster_use_minibatch_kmeans(False) is False
    assert bicluster_use_minibatch_kmeans(True) is True

    assert np.allclose(bicluster_kmeans_centroids(kmeans.cluster_centers_), kmeans.cluster_centers_)
    assert np.array_equal(bicluster_kmeans_labels(kmeans.labels_), kmeans.labels_)
    assert np.array_equal(bicluster_project_cluster_labels(kmeans.labels_), kmeans.labels_)

    assert np.allclose(bicluster_kmeans_centroids(minibatch.cluster_centers_), minibatch.cluster_centers_)
    assert np.array_equal(bicluster_kmeans_labels(minibatch.labels_), minibatch.labels_)
    assert np.array_equal(bicluster_project_cluster_labels(minibatch.labels_), minibatch.labels_)


def test_bicluster_kmeans_shell_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_kmeans_shell import (
        bicluster_kmeans_centroids,
        bicluster_kmeans_labels,
        bicluster_project_cluster_labels,
        bicluster_use_minibatch_kmeans,
    )

    with pytest.raises(ViolationError):
        bicluster_use_minibatch_kmeans(1)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        bicluster_kmeans_centroids(np.array([[0.0, np.nan]], dtype=np.float64))

    with pytest.raises(ViolationError):
        bicluster_kmeans_labels(np.array([0, -1], dtype=np.int64))

    with pytest.raises(ViolationError):
        bicluster_project_cluster_labels(np.array([-1, 0], dtype=np.int64))
