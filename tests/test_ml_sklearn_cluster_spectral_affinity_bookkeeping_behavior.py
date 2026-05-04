from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
from unittest.mock import patch


def test_spectral_affinity_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_affinity_bookkeeping import (
        spectral_fit_pairwise_kernel_params,
        spectral_fit_symmetric_connectivity,
        spectral_fit_use_nearest_neighbors,
        spectral_fit_use_pairwise_kernel_hyperparameters,
        spectral_fit_use_precomputed_affinity,
        spectral_fit_use_precomputed_nearest_neighbors,
    )

    assert callable(spectral_fit_use_nearest_neighbors)
    assert callable(spectral_fit_use_precomputed_nearest_neighbors)
    assert callable(spectral_fit_use_precomputed_affinity)
    assert callable(spectral_fit_use_pairwise_kernel_hyperparameters)
    assert callable(spectral_fit_pairwise_kernel_params)
    assert callable(spectral_fit_symmetric_connectivity)


def test_spectral_affinity_bookkeeping_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_affinity_bookkeeping import (
        spectral_fit_pairwise_kernel_params,
        spectral_fit_symmetric_connectivity,
        spectral_fit_use_nearest_neighbors,
        spectral_fit_use_pairwise_kernel_hyperparameters,
        spectral_fit_use_precomputed_affinity,
        spectral_fit_use_precomputed_nearest_neighbors,
    )

    assert spectral_fit_use_nearest_neighbors("nearest_neighbors") is True
    assert spectral_fit_use_nearest_neighbors("rbf") is False
    assert spectral_fit_use_precomputed_nearest_neighbors("precomputed_nearest_neighbors") is True
    assert spectral_fit_use_precomputed_nearest_neighbors("precomputed") is False
    assert spectral_fit_use_precomputed_affinity("precomputed") is True
    assert spectral_fit_use_precomputed_affinity("nearest_neighbors") is False
    assert spectral_fit_use_pairwise_kernel_hyperparameters("rbf") is True
    assert spectral_fit_use_pairwise_kernel_hyperparameters(lambda X, Y=None: X) is False

    assert spectral_fit_pairwise_kernel_params(None, True, 0.5, 3, 1.25) == {
        "gamma": 0.5,
        "degree": 3,
        "coef0": 1.25,
    }
    assert spectral_fit_pairwise_kernel_params({"existing": 7}, False, 0.5, 3, 1.25) == {
        "existing": 7,
    }

    dense = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64)
    dense_expected = 0.5 * (dense + dense.T)
    assert np.allclose(spectral_fit_symmetric_connectivity(dense), dense_expected)

    sparse = sp.csr_matrix(np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64))
    sparse_result = spectral_fit_symmetric_connectivity(sparse)
    assert sp.issparse(sparse_result)
    assert np.allclose(sparse_result.toarray(), dense_expected)


def test_spectral_affinity_bookkeeping_matches_sklearn_branch_outputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_affinity_bookkeeping import (
        spectral_fit_pairwise_kernel_params,
        spectral_fit_symmetric_connectivity,
        spectral_fit_use_pairwise_kernel_hyperparameters,
    )

    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    pairwise_matrix = np.array(
        [[1.0, 0.25, 0.5], [0.25, 1.0, 0.75], [0.5, 0.75, 1.0]],
        dtype=np.float64,
    )
    embedding = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64)
    labels = np.array([0, 1, 0], dtype=np.int64)

    estimator = SpectralClustering(
        n_clusters=2,
        affinity="poly",
        gamma=0.5,
        degree=3,
        coef0=1.25,
        assign_labels="cluster_qr",
    )
    expected_params = spectral_fit_pairwise_kernel_params(
        estimator.kernel_params,
        spectral_fit_use_pairwise_kernel_hyperparameters(estimator.affinity),
        estimator.gamma,
        estimator.degree,
        estimator.coef0,
    )

    with (
        patch("sklearn.cluster._spectral.pairwise_kernels", autospec=True, return_value=pairwise_matrix) as pairwise_mock,
        patch("sklearn.cluster._spectral._spectral_embedding", autospec=True, return_value=embedding),
        patch("sklearn.cluster._spectral.cluster_qr", autospec=True, return_value=labels),
    ):
        fitted = estimator.fit(X)

    kwargs = pairwise_mock.call_args.kwargs
    assert kwargs["metric"] == estimator.affinity
    assert kwargs["filter_params"] is True
    assert {key: kwargs[key] for key in expected_params} == expected_params
    assert np.array_equal(fitted.labels_, labels)

    connectivity = sp.csr_matrix(
        np.array(
            [[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 2.0, 1.0]],
            dtype=np.float64,
        )
    )
    nn_estimator = SpectralClustering(
        n_clusters=2,
        affinity="nearest_neighbors",
        n_neighbors=1,
        assign_labels="cluster_qr",
    )
    expected_affinity = spectral_fit_symmetric_connectivity(connectivity)

    with (
        patch("sklearn.cluster._spectral.kneighbors_graph", autospec=True, return_value=connectivity),
        patch("sklearn.cluster._spectral._spectral_embedding", autospec=True, return_value=embedding),
        patch("sklearn.cluster._spectral.cluster_qr", autospec=True, return_value=labels),
    ):
        fitted_nn = nn_estimator.fit(X)

    assert sp.issparse(fitted_nn.affinity_matrix_)
    assert np.allclose(fitted_nn.affinity_matrix_.toarray(), expected_affinity.toarray())


def test_spectral_affinity_bookkeeping_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_affinity_bookkeeping import (
        spectral_fit_pairwise_kernel_params,
        spectral_fit_symmetric_connectivity,
        spectral_fit_use_nearest_neighbors,
        spectral_fit_use_pairwise_kernel_hyperparameters,
        spectral_fit_use_precomputed_affinity,
        spectral_fit_use_precomputed_nearest_neighbors,
    )

    with pytest.raises(Exception):
        spectral_fit_use_nearest_neighbors("")

    with pytest.raises(Exception):
        spectral_fit_use_precomputed_nearest_neighbors("")

    with pytest.raises(Exception):
        spectral_fit_use_precomputed_affinity("")

    with pytest.raises(Exception):
        spectral_fit_use_pairwise_kernel_hyperparameters(7)

    with pytest.raises(Exception):
        spectral_fit_pairwise_kernel_params({1: "bad"}, True, 0.5, 3, 1.25)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        spectral_fit_pairwise_kernel_params(None, True, float("nan"), 3, 1.25)

    with pytest.raises(Exception):
        spectral_fit_symmetric_connectivity(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))
