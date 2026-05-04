from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pytest


def test_spectral_pairwise_kernel_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_pairwise_kernel_shell import (
        spectral_fit_pairwise_affinity_matrix,
        spectral_fit_pairwise_kernel_kwargs,
    )

    assert callable(spectral_fit_pairwise_kernel_kwargs)
    assert callable(spectral_fit_pairwise_affinity_matrix)


def test_spectral_pairwise_kernel_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_pairwise_kernel_shell import (
        spectral_fit_pairwise_affinity_matrix,
        spectral_fit_pairwise_kernel_kwargs,
    )

    def affinity(x: np.ndarray, y: np.ndarray) -> float:
        del x
        del y
        return 0.0

    params = {"gamma": 0.75, "degree": 4.0, "coef0": 1.5}
    assert spectral_fit_pairwise_kernel_kwargs("rbf", params) == {
        "metric": "rbf",
        "filter_params": True,
        **params,
    }
    callable_kwargs = spectral_fit_pairwise_kernel_kwargs(affinity, {"bandwidth": 2.0})
    assert callable_kwargs["metric"] is affinity
    assert callable_kwargs["filter_params"] is True
    assert callable_kwargs["bandwidth"] == 2.0

    affinity_matrix = np.array(
        [[1.0, 0.5, 0.25], [0.5, 1.0, 0.125], [0.25, 0.125, 1.0]],
        dtype=np.float64,
    )
    assert np.array_equal(
        spectral_fit_pairwise_affinity_matrix(affinity_matrix),
        affinity_matrix,
    )


def test_spectral_pairwise_kernel_shell_matches_sklearn_fit_call() -> None:
    from sklearn.cluster import SpectralClustering

    from sciona.atoms.ml.sklearn.cluster.spectral_affinity_bookkeeping import (
        spectral_fit_pairwise_kernel_params,
        spectral_fit_use_pairwise_kernel_hyperparameters,
    )
    from sciona.atoms.ml.sklearn.cluster.spectral_pairwise_kernel_shell import (
        spectral_fit_pairwise_affinity_matrix,
        spectral_fit_pairwise_kernel_kwargs,
    )

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    affinity_matrix = np.array(
        [[1.0, 0.5, 0.25], [0.5, 1.0, 0.125], [0.25, 0.125, 1.0]],
        dtype=np.float64,
    )
    embedding = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64)
    labels = np.array([0, 1, 0], dtype=np.int64)

    estimator = SpectralClustering(
        n_clusters=2,
        affinity="rbf",
        gamma=0.75,
        degree=4,
        coef0=1.5,
        assign_labels="cluster_qr",
    )
    with (
        patch("sklearn.cluster._spectral.pairwise_kernels", autospec=True, return_value=affinity_matrix) as pairwise_mock,
        patch("sklearn.cluster._spectral._spectral_embedding", autospec=True, return_value=embedding),
        patch("sklearn.cluster._spectral.cluster_qr", autospec=True, return_value=labels),
    ):
        fitted = estimator.fit(X)

    expected_params = spectral_fit_pairwise_kernel_params(
        estimator.kernel_params,
        spectral_fit_use_pairwise_kernel_hyperparameters(estimator.affinity),
        estimator.gamma,
        estimator.degree,
        estimator.coef0,
    )
    assert np.array_equal(pairwise_mock.call_args.args[0], X)
    assert pairwise_mock.call_args.kwargs == spectral_fit_pairwise_kernel_kwargs(
        estimator.affinity,
        expected_params,
    )
    assert np.array_equal(
        fitted.affinity_matrix_,
        spectral_fit_pairwise_affinity_matrix(affinity_matrix),
    )


def test_spectral_pairwise_kernel_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_pairwise_kernel_shell import (
        spectral_fit_pairwise_affinity_matrix,
        spectral_fit_pairwise_kernel_kwargs,
    )

    with pytest.raises(Exception):
        spectral_fit_pairwise_kernel_kwargs("", {"gamma": 1.0})

    with pytest.raises(Exception):
        spectral_fit_pairwise_kernel_kwargs("rbf", {1: "bad"})  # type: ignore[arg-type]

    with pytest.raises(Exception):
        spectral_fit_pairwise_affinity_matrix(np.array([[1.0, np.nan]], dtype=np.float64))

