from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def test_bicluster_kmeans_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_kmeans_callback_shell import (
        bicluster_kmeans_kwargs,
        bicluster_minibatch_kmeans_kwargs,
    )

    assert callable(bicluster_kmeans_kwargs)
    assert callable(bicluster_minibatch_kmeans_kwargs)


def test_bicluster_kmeans_callback_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_kmeans_callback_shell import (
        bicluster_kmeans_kwargs,
        bicluster_minibatch_kmeans_kwargs,
    )

    rng = np.random.RandomState(7)
    init = "k-means++"
    expected = {
        "init": init,
        "n_init": 5,
        "random_state": rng,
    }
    assert bicluster_kmeans_kwargs(3, init, 5, rng) == expected
    assert bicluster_minibatch_kmeans_kwargs(3, init, 5, rng) == expected


def test_bicluster_kmeans_callback_shell_matches_base_spectral_calls() -> None:
    from sklearn.cluster import SpectralBiclustering

    from sciona.atoms.ml.sklearn.cluster.bicluster_kmeans_callback_shell import (
        bicluster_kmeans_kwargs,
        bicluster_minibatch_kmeans_kwargs,
    )

    data = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)

    model = SpectralBiclustering(n_clusters=2, mini_batch=False, init="random", n_init=4, random_state=13)
    with patch("sklearn.cluster._bicluster.KMeans", autospec=True) as kmeans_mock:
        kmeans_mock.return_value.cluster_centers_ = np.array([[0.0], [1.0]], dtype=np.float64)
        kmeans_mock.return_value.labels_ = np.array([0, 1, 1], dtype=np.int64)
        model._k_means(data, 2)
    assert kmeans_mock.call_args.args == (2,)
    assert kmeans_mock.call_args.kwargs == bicluster_kmeans_kwargs(
        2,
        model.init,
        model.n_init,
        model.random_state,
    )

    mini = SpectralBiclustering(n_clusters=2, mini_batch=True, init="random", n_init=4, random_state=13)
    with patch("sklearn.cluster._bicluster.MiniBatchKMeans", autospec=True) as minibatch_mock:
        minibatch_mock.return_value.cluster_centers_ = np.array([[0.0], [1.0]], dtype=np.float64)
        minibatch_mock.return_value.labels_ = np.array([0, 1, 1], dtype=np.int64)
        mini._k_means(data, 2)
    assert minibatch_mock.call_args.args == (2,)
    assert minibatch_mock.call_args.kwargs == bicluster_minibatch_kmeans_kwargs(
        2,
        mini.init,
        mini.n_init,
        mini.random_state,
    )


def test_bicluster_kmeans_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_kmeans_callback_shell import (
        bicluster_kmeans_kwargs,
        bicluster_minibatch_kmeans_kwargs,
    )

    with pytest.raises(Exception):
        bicluster_kmeans_kwargs(0, "k-means++", 5, None)

    with pytest.raises(Exception):
        bicluster_minibatch_kmeans_kwargs(2, "", 5, None)
