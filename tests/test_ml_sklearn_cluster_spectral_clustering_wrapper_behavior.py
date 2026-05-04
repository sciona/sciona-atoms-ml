from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import SpectralClustering, spectral_clustering


def test_spectral_clustering_wrapper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_clustering_wrapper import (
        spectral_clustering_precomputed_affinity,
        spectral_clustering_return_labels,
    )

    assert callable(spectral_clustering_precomputed_affinity)
    assert callable(spectral_clustering_return_labels)


def test_spectral_clustering_wrapper_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_clustering_wrapper import (
        spectral_clustering_precomputed_affinity,
        spectral_clustering_return_labels,
    )

    affinity = np.array(
        [
            [1.0, 0.8, 0.1],
            [0.8, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ],
        dtype=np.float64,
    )

    labels = spectral_clustering(
        affinity,
        n_clusters=2,
        random_state=0,
        n_init=1,
        assign_labels="cluster_qr",
    )

    clusterer = SpectralClustering(
        n_clusters=2,
        n_components=None,
        eigen_solver=None,
        random_state=0,
        n_init=1,
        affinity=spectral_clustering_precomputed_affinity(None),
        eigen_tol="auto",
        assign_labels="cluster_qr",
        verbose=False,
    ).fit(affinity)

    assert clusterer.affinity == "precomputed"
    assert np.array_equal(spectral_clustering_return_labels(clusterer.labels_), clusterer.labels_)
    assert np.array_equal(labels, clusterer.labels_)


def test_spectral_clustering_wrapper_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_clustering_wrapper import (
        spectral_clustering_precomputed_affinity,
        spectral_clustering_return_labels,
    )

    with pytest.raises(Exception):
        spectral_clustering_precomputed_affinity("")

    with pytest.raises(Exception):
        spectral_clustering_return_labels(np.array([[1, 2]], dtype=np.int64))
