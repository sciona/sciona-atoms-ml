from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch


def test_spectral_label_selection_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_label_selection import (
        spectral_fit_selected_labels,
        spectral_fit_use_discretize,
    )

    assert callable(spectral_fit_use_discretize)
    assert callable(spectral_fit_selected_labels)


def test_spectral_label_selection_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_fit_bookkeeping import (
        spectral_fit_use_cluster_qr,
        spectral_fit_use_kmeans,
    )
    from sciona.atoms.ml.sklearn.cluster.spectral_label_selection import (
        spectral_fit_selected_labels,
        spectral_fit_use_discretize,
    )

    kmeans_labels = np.array([1, 0, 1], dtype=np.int64)
    cluster_qr_labels = np.array([0, 1, 1], dtype=np.int64)
    discretize_labels = np.array([2, 2, 0], dtype=np.int64)

    assert spectral_fit_use_discretize("discretize") is True
    assert spectral_fit_use_discretize("kmeans") is False
    assert spectral_fit_use_discretize("cluster_qr") is False
    assert spectral_fit_use_discretize("unexpected") is True

    assert np.array_equal(
        spectral_fit_selected_labels(
            spectral_fit_use_kmeans("kmeans"),
            spectral_fit_use_cluster_qr("kmeans"),
            kmeans_labels,
            cluster_qr_labels,
            discretize_labels,
        ),
        kmeans_labels,
    )
    assert np.array_equal(
        spectral_fit_selected_labels(
            spectral_fit_use_kmeans("cluster_qr"),
            spectral_fit_use_cluster_qr("cluster_qr"),
            kmeans_labels,
            cluster_qr_labels,
            discretize_labels,
        ),
        cluster_qr_labels,
    )
    assert np.array_equal(
        spectral_fit_selected_labels(
            spectral_fit_use_kmeans("discretize"),
            spectral_fit_use_cluster_qr("discretize"),
            kmeans_labels,
            cluster_qr_labels,
            discretize_labels,
        ),
        discretize_labels,
    )


def test_spectral_label_selection_matches_sklearn_branch_outputs() -> None:
    from sklearn.cluster import SpectralClustering

    from sciona.atoms.ml.sklearn.cluster.spectral_fit_bookkeeping import (
        spectral_fit_use_cluster_qr,
        spectral_fit_use_kmeans,
    )
    from sciona.atoms.ml.sklearn.cluster.spectral_label_selection import (
        spectral_fit_selected_labels,
    )

    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    embedding = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64)
    kmeans_labels = np.array([1, 0, 1], dtype=np.int64)
    cluster_qr_labels = np.array([0, 1, 1], dtype=np.int64)
    discretize_labels = np.array([2, 2, 0], dtype=np.int64)

    estimator = SpectralClustering(n_clusters=2, affinity="precomputed", assign_labels="discretize")
    with (
        patch("sklearn.cluster._spectral._spectral_embedding", autospec=True, return_value=embedding),
        patch("sklearn.cluster._spectral.discretize", autospec=True, return_value=discretize_labels),
    ):
        fitted = estimator.fit(np.eye(3, dtype=np.float64))

    expected = spectral_fit_selected_labels(
        spectral_fit_use_kmeans(estimator.assign_labels),
        spectral_fit_use_cluster_qr(estimator.assign_labels),
        kmeans_labels,
        cluster_qr_labels,
        discretize_labels,
    )
    assert np.array_equal(fitted.labels_, expected)

    estimator_cluster_qr = SpectralClustering(n_clusters=2, affinity="precomputed", assign_labels="cluster_qr")
    with (
        patch("sklearn.cluster._spectral._spectral_embedding", autospec=True, return_value=embedding),
        patch("sklearn.cluster._spectral.cluster_qr", autospec=True, return_value=cluster_qr_labels),
    ):
        fitted_cluster_qr = estimator_cluster_qr.fit(np.eye(3, dtype=np.float64))

    expected_cluster_qr = spectral_fit_selected_labels(
        spectral_fit_use_kmeans(estimator_cluster_qr.assign_labels),
        spectral_fit_use_cluster_qr(estimator_cluster_qr.assign_labels),
        kmeans_labels,
        cluster_qr_labels,
        discretize_labels,
    )
    assert np.array_equal(fitted_cluster_qr.labels_, expected_cluster_qr)


def test_spectral_label_selection_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_label_selection import (
        spectral_fit_selected_labels,
        spectral_fit_use_discretize,
    )

    with pytest.raises(Exception):
        spectral_fit_use_discretize("")

    with pytest.raises(Exception):
        spectral_fit_selected_labels(
            True,
            True,
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        )

    with pytest.raises(Exception):
        spectral_fit_selected_labels(
            False,
            False,
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        )
