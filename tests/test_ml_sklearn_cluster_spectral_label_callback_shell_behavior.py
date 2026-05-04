from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch


def test_spectral_label_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_label_callback_shell import (
        spectral_fit_discretize_kwargs,
        spectral_fit_kmeans_kwargs,
        spectral_fit_kmeans_output_labels,
    )

    assert callable(spectral_fit_kmeans_kwargs)
    assert callable(spectral_fit_kmeans_output_labels)
    assert callable(spectral_fit_discretize_kwargs)


def test_spectral_label_callback_shell_matches_source_logic() -> None:
    from sklearn.utils import check_random_state

    from sciona.atoms.ml.sklearn.cluster.spectral_label_callback_shell import (
        spectral_fit_discretize_kwargs,
        spectral_fit_kmeans_kwargs,
        spectral_fit_kmeans_output_labels,
    )

    rng = check_random_state(7)
    kmeans_kwargs = spectral_fit_kmeans_kwargs(3, 10, False, rng)
    assert kmeans_kwargs == {
        "random_state": rng,
        "n_init": 10,
        "verbose": False,
    }
    assert spectral_fit_kmeans_output_labels(np.array([1, 0, 1], dtype=np.int64)).tolist() == [1, 0, 1]
    assert spectral_fit_discretize_kwargs(rng) == {"random_state": rng}


def test_spectral_label_callback_shell_matches_sklearn_calls() -> None:
    from sklearn.cluster import SpectralClustering
    from sklearn.utils import check_random_state

    from sciona.atoms.ml.sklearn.cluster.spectral_embedding_call_shell import spectral_fit_embedding_random_state
    from sciona.atoms.ml.sklearn.cluster.spectral_label_callback_shell import (
        spectral_fit_discretize_kwargs,
        spectral_fit_kmeans_kwargs,
        spectral_fit_kmeans_output_labels,
    )

    X = np.eye(3, dtype=np.float64)
    embedding = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64)
    kmeans_labels = np.array([1, 0, 1], dtype=np.int64)
    cluster_qr_labels = np.array([0, 1, 0], dtype=np.int64)
    discretize_labels = np.array([2, 2, 0], dtype=np.int64)

    estimator = SpectralClustering(
        n_clusters=2,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=11,
        n_init=5,
        verbose=False,
    )
    with (
        patch("sklearn.cluster._spectral._spectral_embedding", autospec=True, return_value=embedding),
        patch("sklearn.cluster._spectral.k_means", autospec=True, return_value=(np.zeros((2, 2)), kmeans_labels, 0.0)) as kmeans_mock,
    ):
        estimator.fit(X)
    expected_rng = spectral_fit_embedding_random_state(estimator.random_state)
    expected_kmeans_kwargs = spectral_fit_kmeans_kwargs(estimator.n_clusters, estimator.n_init, estimator.verbose, expected_rng)
    assert kmeans_mock.call_args.args[1] == estimator.n_clusters
    assert kmeans_mock.call_args.kwargs["n_init"] == expected_kmeans_kwargs["n_init"]
    assert kmeans_mock.call_args.kwargs["verbose"] == expected_kmeans_kwargs["verbose"]
    assert np.array_equal(estimator.labels_, spectral_fit_kmeans_output_labels(kmeans_labels))

    estimator_discretize = SpectralClustering(
        n_clusters=2,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=13,
    )
    with (
        patch("sklearn.cluster._spectral._spectral_embedding", autospec=True, return_value=embedding),
        patch("sklearn.cluster._spectral.discretize", autospec=True, return_value=discretize_labels) as discretize_mock,
    ):
        estimator_discretize.fit(X)
    expected_discretize_kwargs = spectral_fit_discretize_kwargs(spectral_fit_embedding_random_state(estimator_discretize.random_state))
    assert discretize_mock.call_args.kwargs.keys() == expected_discretize_kwargs.keys()
    assert isinstance(discretize_mock.call_args.kwargs["random_state"], np.random.RandomState)
    assert np.array_equal(estimator_discretize.labels_, discretize_labels)
    assert np.array_equal(cluster_qr_labels, np.array([0, 1, 0], dtype=np.int64))


def test_spectral_label_callback_shell_contracts() -> None:
    from sklearn.utils import check_random_state

    from sciona.atoms.ml.sklearn.cluster.spectral_label_callback_shell import (
        spectral_fit_discretize_kwargs,
        spectral_fit_kmeans_kwargs,
        spectral_fit_kmeans_output_labels,
    )

    rng = check_random_state(0)
    with pytest.raises(Exception):
        spectral_fit_kmeans_kwargs(0, 5, False, rng)

    with pytest.raises(Exception):
        spectral_fit_kmeans_kwargs(2, 0, False, rng)

    with pytest.raises(Exception):
        spectral_fit_kmeans_kwargs(2, 5, False, 7)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        spectral_fit_kmeans_output_labels(np.array([[0, 1]], dtype=np.int64))

    with pytest.raises(Exception):
        spectral_fit_discretize_kwargs(7)  # type: ignore[arg-type]
