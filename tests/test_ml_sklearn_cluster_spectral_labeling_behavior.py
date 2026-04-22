from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster._spectral import cluster_qr, discretize


def _embedding() -> np.ndarray:
    return np.array(
        [
            [0.92, 0.05, 0.10],
            [0.85, 0.12, 0.05],
            [0.08, 0.96, 0.10],
            [0.12, 0.88, 0.14],
            [0.05, 0.20, 0.90],
            [0.10, 0.14, 0.86],
        ],
        dtype=np.float64,
    )


def test_spectral_labeling_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_labeling import (
        spectral_cluster_qr_labels,
        spectral_discretize_labels,
    )

    assert callable(spectral_cluster_qr_labels)
    assert callable(spectral_discretize_labels)


def test_spectral_cluster_qr_labels_match_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_labeling import spectral_cluster_qr_labels

    vectors = _embedding()
    result = spectral_cluster_qr_labels(vectors)
    expected = cluster_qr(vectors)

    assert np.array_equal(result, expected)
    assert result.shape == (vectors.shape[0],)
    assert set(result.tolist()) == {0, 1, 2}


def test_spectral_cluster_qr_labels_handles_two_component_embedding() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_labeling import spectral_cluster_qr_labels

    vectors = np.array([[0.9, 0.1], [0.7, 0.2], [0.2, 0.9], [0.1, 0.8]], dtype=np.float64)
    assert np.array_equal(spectral_cluster_qr_labels(vectors), cluster_qr(vectors))


def test_spectral_discretize_labels_match_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_labeling import spectral_discretize_labels

    vectors = _embedding()
    original = vectors.copy()
    result = spectral_discretize_labels(vectors, random_state=4, n_iter_max=30)
    expected = discretize(vectors, random_state=4, n_iter_max=30, copy=True)

    assert np.array_equal(result, expected)
    assert np.array_equal(vectors, original)
    assert result.shape == (vectors.shape[0],)
    assert set(result.tolist()) == {0, 1, 2}


def test_spectral_discretize_labels_matches_sklearn_for_rotated_embedding() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_labeling import spectral_discretize_labels

    vectors = np.array(
        [
            [0.78, 0.15, 0.20],
            [0.72, 0.20, 0.12],
            [0.20, 0.78, 0.16],
            [0.18, 0.71, 0.25],
            [0.16, 0.22, 0.74],
            [0.25, 0.18, 0.70],
        ],
        dtype=np.float64,
    )
    assert np.array_equal(
        spectral_discretize_labels(vectors, random_state=11, max_svd_restarts=10, n_iter_max=40),
        discretize(vectors, random_state=11, max_svd_restarts=10, n_iter_max=40, copy=True),
    )


def test_contracts_reject_invalid_spectral_labeling_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_labeling import (
        spectral_cluster_qr_labels,
        spectral_discretize_labels,
    )

    with pytest.raises(ViolationError):
        spectral_cluster_qr_labels(np.ones((2, 3), dtype=np.float64))

    with pytest.raises(ViolationError):
        spectral_cluster_qr_labels(np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        spectral_discretize_labels(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        spectral_discretize_labels(_embedding(), max_svd_restarts=0)

    with pytest.raises(ViolationError):
        spectral_discretize_labels(_embedding(), n_iter_max=0)
