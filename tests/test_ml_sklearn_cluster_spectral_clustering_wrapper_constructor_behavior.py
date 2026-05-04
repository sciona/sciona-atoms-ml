from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def test_spectral_clustering_wrapper_constructor_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_clustering_wrapper_constructor import (
        spectral_clustering_constructor_kwargs,
    )

    assert callable(spectral_clustering_constructor_kwargs)


def test_spectral_clustering_wrapper_constructor_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_clustering_wrapper_constructor import (
        spectral_clustering_constructor_kwargs,
    )

    rng = np.random.RandomState(7)
    assert spectral_clustering_constructor_kwargs(
        n_clusters=3,
        n_components=None,
        eigen_solver="arpack",
        random_state=rng,
        n_init=5,
        eigen_tol="auto",
        assign_labels="cluster_qr",
        verbose=False,
    ) == {
        "n_clusters": 3,
        "n_components": None,
        "eigen_solver": "arpack",
        "random_state": rng,
        "n_init": 5,
        "affinity": "precomputed",
        "eigen_tol": "auto",
        "assign_labels": "cluster_qr",
        "verbose": False,
    }


def test_spectral_clustering_wrapper_constructor_matches_sklearn_call() -> None:
    from sklearn.cluster._spectral import spectral_clustering

    from sciona.atoms.ml.sklearn.cluster.spectral_clustering_wrapper_constructor import (
        spectral_clustering_constructor_kwargs,
    )

    affinity = np.eye(3, dtype=np.float64)
    labels = np.array([0, 1, 0], dtype=np.int64)
    random_state = np.random.RandomState(11)

    with patch("sklearn.cluster._spectral.SpectralClustering", autospec=True) as clusterer_mock:
        clusterer_mock.return_value.fit.return_value.labels_ = labels
        result = spectral_clustering(
            affinity,
            n_clusters=2,
            n_components=2,
            eigen_solver="arpack",
            random_state=random_state,
            n_init=4,
            eigen_tol=0.0,
            assign_labels="discretize",
            verbose=1,
        )

    assert clusterer_mock.call_args.kwargs == spectral_clustering_constructor_kwargs(
        n_clusters=2,
        n_components=2,
        eigen_solver="arpack",
        random_state=random_state,
        n_init=4,
        eigen_tol=0.0,
        assign_labels="discretize",
        verbose=1,
    )
    assert np.array_equal(clusterer_mock.return_value.fit.call_args.args[0], affinity)
    assert np.array_equal(result, labels)


def test_spectral_clustering_wrapper_constructor_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_clustering_wrapper_constructor import (
        spectral_clustering_constructor_kwargs,
    )

    with pytest.raises(Exception):
        spectral_clustering_constructor_kwargs(0, None, None)

    with pytest.raises(Exception):
        spectral_clustering_constructor_kwargs(2, 0, None)

    with pytest.raises(Exception):
        spectral_clustering_constructor_kwargs(2, None, "")
