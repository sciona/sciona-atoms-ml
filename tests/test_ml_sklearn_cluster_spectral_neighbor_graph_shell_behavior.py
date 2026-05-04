from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch


def test_spectral_neighbor_graph_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_neighbor_graph_shell import (
        spectral_fit_kneighbors_graph_kwargs,
        spectral_fit_precomputed_kneighbors_graph_mode,
        spectral_fit_precomputed_neighbor_estimator_kwargs,
        spectral_fit_precomputed_neighbor_metric,
    )

    assert callable(spectral_fit_kneighbors_graph_kwargs)
    assert callable(spectral_fit_precomputed_neighbor_metric)
    assert callable(spectral_fit_precomputed_neighbor_estimator_kwargs)
    assert callable(spectral_fit_precomputed_kneighbors_graph_mode)


def test_spectral_neighbor_graph_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_neighbor_graph_shell import (
        spectral_fit_kneighbors_graph_kwargs,
        spectral_fit_precomputed_kneighbors_graph_mode,
        spectral_fit_precomputed_neighbor_estimator_kwargs,
        spectral_fit_precomputed_neighbor_metric,
    )

    assert spectral_fit_kneighbors_graph_kwargs(7, None) == {
        "n_neighbors": 7,
        "include_self": True,
        "n_jobs": None,
    }
    assert spectral_fit_precomputed_neighbor_metric(None) == "precomputed"
    assert spectral_fit_precomputed_neighbor_estimator_kwargs(5, 3) == {
        "n_neighbors": 5,
        "n_jobs": 3,
        "metric": "precomputed",
    }
    assert spectral_fit_precomputed_kneighbors_graph_mode(None) == "connectivity"


def test_spectral_neighbor_graph_shell_matches_sklearn_calls() -> None:
    from sklearn.cluster import SpectralClustering
    from scipy import sparse

    from sciona.atoms.ml.sklearn.cluster.spectral_neighbor_graph_shell import (
        spectral_fit_kneighbors_graph_kwargs,
        spectral_fit_precomputed_kneighbors_graph_mode,
        spectral_fit_precomputed_neighbor_estimator_kwargs,
    )

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    connectivity = sparse.csr_matrix(np.eye(3, dtype=np.float64))
    embedding = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64)
    labels = np.array([0, 1, 0], dtype=np.int64)

    nearest = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=2, assign_labels="cluster_qr")
    with (
        patch("sklearn.cluster._spectral.kneighbors_graph", autospec=True, return_value=connectivity) as kneighbors_mock,
        patch("sklearn.cluster._spectral._spectral_embedding", autospec=True, return_value=embedding),
        patch("sklearn.cluster._spectral.cluster_qr", autospec=True, return_value=labels),
    ):
        nearest.fit(X)
    assert kneighbors_mock.call_args.kwargs == spectral_fit_kneighbors_graph_kwargs(nearest.n_neighbors, nearest.n_jobs)

    precomputed = SpectralClustering(
        n_clusters=2,
        affinity="precomputed_nearest_neighbors",
        n_neighbors=2,
        assign_labels="cluster_qr",
    )
    distances = np.array([[0.0, 1.0, 1.5], [1.0, 0.0, 1.2], [1.5, 1.2, 0.0]], dtype=np.float64)
    with (
        patch("sklearn.cluster._spectral.NearestNeighbors", autospec=True) as nn_mock,
        patch("sklearn.cluster._spectral._spectral_embedding", autospec=True, return_value=embedding),
        patch("sklearn.cluster._spectral.cluster_qr", autospec=True, return_value=labels),
    ):
        nn_mock.return_value.fit.return_value.kneighbors_graph.return_value = connectivity
        precomputed.fit(distances)
    assert nn_mock.call_args.kwargs == spectral_fit_precomputed_neighbor_estimator_kwargs(precomputed.n_neighbors, precomputed.n_jobs)
    assert nn_mock.return_value.fit.return_value.kneighbors_graph.call_args.kwargs == {
        "X": distances,
        "mode": spectral_fit_precomputed_kneighbors_graph_mode(None),
    }


def test_spectral_neighbor_graph_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_neighbor_graph_shell import (
        spectral_fit_kneighbors_graph_kwargs,
        spectral_fit_precomputed_kneighbors_graph_mode,
        spectral_fit_precomputed_neighbor_estimator_kwargs,
        spectral_fit_precomputed_neighbor_metric,
    )

    with pytest.raises(Exception):
        spectral_fit_kneighbors_graph_kwargs(0, None)

    with pytest.raises(Exception):
        spectral_fit_kneighbors_graph_kwargs(2, "auto")  # type: ignore[arg-type]

    with pytest.raises(Exception):
        spectral_fit_precomputed_neighbor_metric(1)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        spectral_fit_precomputed_neighbor_estimator_kwargs(0, None)

    with pytest.raises(Exception):
        spectral_fit_precomputed_kneighbors_graph_mode(1)  # type: ignore[arg-type]
