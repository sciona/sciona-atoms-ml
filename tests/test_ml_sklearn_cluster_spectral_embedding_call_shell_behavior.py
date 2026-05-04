from __future__ import annotations

import numpy as np
import pytest


def test_spectral_embedding_call_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_embedding_call_shell import (
        spectral_fit_embedding_call_kwargs,
        spectral_fit_embedding_drop_first,
        spectral_fit_embedding_random_state,
    )

    assert callable(spectral_fit_embedding_random_state)
    assert callable(spectral_fit_embedding_drop_first)
    assert callable(spectral_fit_embedding_call_kwargs)


def test_spectral_embedding_call_shell_matches_source_logic() -> None:
    from sklearn.utils import check_random_state

    from sciona.atoms.ml.sklearn.cluster.spectral_embedding_call_shell import (
        spectral_fit_embedding_call_kwargs,
        spectral_fit_embedding_drop_first,
        spectral_fit_embedding_random_state,
    )

    rng = spectral_fit_embedding_random_state(7)
    expected = check_random_state(7)
    assert np.array_equal(
        rng.randint(np.iinfo(np.int32).max, size=4),
        expected.randint(np.iinfo(np.int32).max, size=4),
    )
    assert spectral_fit_embedding_drop_first(True) is False

    kwargs = spectral_fit_embedding_call_kwargs(3, "arpack", 11, "auto")
    expected_rng = check_random_state(11)
    assert kwargs["n_components"] == 3
    assert kwargs["eigen_solver"] == "arpack"
    assert kwargs["eigen_tol"] == "auto"
    assert kwargs["drop_first"] is False
    assert np.array_equal(
        kwargs["random_state"].randint(np.iinfo(np.int32).max, size=4),
        expected_rng.randint(np.iinfo(np.int32).max, size=4),
    )


def test_spectral_embedding_call_shell_matches_sklearn_fit_call() -> None:
    from unittest.mock import patch

    from sklearn.cluster import SpectralClustering

    from sciona.atoms.ml.sklearn.cluster.spectral_embedding_call_shell import (
        spectral_fit_embedding_call_kwargs,
    )

    X = np.eye(3, dtype=np.float64)
    embedding = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64)
    labels = np.array([0, 1, 0], dtype=np.int64)
    estimator = SpectralClustering(
        n_clusters=2,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=13,
        eigen_solver="arpack",
        eigen_tol="auto",
    )

    with (
        patch("sklearn.cluster._spectral._spectral_embedding", autospec=True, return_value=embedding) as embedding_mock,
        patch("sklearn.cluster._spectral.cluster_qr", autospec=True, return_value=labels),
    ):
        estimator.fit(X)

    kwargs = embedding_mock.call_args.kwargs
    expected = spectral_fit_embedding_call_kwargs(
        estimator.n_clusters if estimator.n_components is None else estimator.n_components,
        estimator.eigen_solver,
        estimator.random_state,
        estimator.eigen_tol,
    )
    assert kwargs["n_components"] == expected["n_components"]
    assert kwargs["eigen_solver"] == expected["eigen_solver"]
    assert kwargs["eigen_tol"] == expected["eigen_tol"]
    assert kwargs["drop_first"] is expected["drop_first"]
    assert np.array_equal(
        kwargs["random_state"].randint(np.iinfo(np.int32).max, size=4),
        expected["random_state"].randint(np.iinfo(np.int32).max, size=4),
    )


def test_spectral_embedding_call_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_embedding_call_shell import (
        spectral_fit_embedding_call_kwargs,
        spectral_fit_embedding_drop_first,
        spectral_fit_embedding_random_state,
    )

    with pytest.raises(Exception):
        spectral_fit_embedding_random_state(-1)

    with pytest.raises(Exception):
        spectral_fit_embedding_drop_first("false")  # type: ignore[arg-type]

    with pytest.raises(Exception):
        spectral_fit_embedding_call_kwargs(0, "arpack", 7, "auto")

    with pytest.raises(Exception):
        spectral_fit_embedding_call_kwargs(2, "arpack", -1, "auto")

    with pytest.raises(Exception):
        spectral_fit_embedding_call_kwargs(2, "arpack", 7, -1.0)
