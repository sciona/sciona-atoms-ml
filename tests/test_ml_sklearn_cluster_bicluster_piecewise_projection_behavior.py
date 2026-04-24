from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
from sklearn.cluster import SpectralBiclustering


def test_bicluster_piecewise_projection_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_piecewise_projection import (
        bicluster_piecewise_residual_norms,
        bicluster_piecewise_vector,
        bicluster_project_dense,
        bicluster_select_best_piecewise_vectors,
    )

    assert callable(bicluster_piecewise_vector)
    assert callable(bicluster_piecewise_residual_norms)
    assert callable(bicluster_select_best_piecewise_vectors)
    assert callable(bicluster_project_dense)


def test_bicluster_piecewise_helpers_match_private_best_piecewise_selection() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_piecewise_projection import (
        bicluster_piecewise_residual_norms,
        bicluster_piecewise_vector,
        bicluster_select_best_piecewise_vectors,
    )

    vectors = np.array(
        [
            [0.1, 0.2, 0.15, 0.25],
            [1.0, 1.1, 0.9, 1.2],
            [0.5, 0.6, 0.4, 0.7],
        ],
        dtype=np.float64,
    )
    piecewise_specs = {
        tuple(vectors[0]): (
            np.array([[0.15], [0.22]], dtype=np.float64),
            np.array([0, 1, 0, 1], dtype=np.int64),
        ),
        tuple(vectors[1]): (
            np.array([[1.0], [1.05]], dtype=np.float64),
            np.array([0, 1, 0, 1], dtype=np.int64),
        ),
        tuple(vectors[2]): (
            np.array([[0.45], [0.65]], dtype=np.float64),
            np.array([0, 1, 0, 1], dtype=np.int64),
        ),
    }
    model = SpectralBiclustering(n_clusters=2, n_best=2, random_state=0)

    def fake_k_means(self: SpectralBiclustering, data: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
        del n_clusters
        spec = piecewise_specs[tuple(np.asarray(data, dtype=np.float64).ravel())]
        return spec

    model._k_means = MethodType(fake_k_means, model)
    sklearn_selected = model._fit_best_piecewise(vectors, n_best=2, n_clusters=2)

    piecewise_vectors = np.vstack(
        [
            bicluster_piecewise_vector(*piecewise_specs[tuple(row)])
            for row in vectors
        ]
    )
    residuals = bicluster_piecewise_residual_norms(vectors, piecewise_vectors)
    selected = bicluster_select_best_piecewise_vectors(vectors, residuals, 2)

    assert np.allclose(selected, sklearn_selected)


def test_bicluster_project_dense_matches_private_projection_step() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_piecewise_projection import bicluster_project_dense

    data = np.array(
        [
            [1.0, 2.0, 0.5],
            [0.2, 1.5, 1.1],
            [2.2, 0.1, 0.7],
        ],
        dtype=np.float64,
    )
    vectors = np.array(
        [
            [0.3, 0.7],
            [0.4, 0.2],
            [0.8, 0.1],
        ],
        dtype=np.float64,
    )
    captured: dict[str, np.ndarray] = {}
    model = SpectralBiclustering(n_clusters=2, random_state=0)

    def fake_k_means(self: SpectralBiclustering, projected: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
        del n_clusters
        captured["projected"] = np.asarray(projected, dtype=np.float64)
        return np.zeros((2, projected.shape[1]), dtype=np.float64), np.zeros(projected.shape[0], dtype=np.int64)

    model._k_means = MethodType(fake_k_means, model)
    model._project_and_cluster(data, vectors, 2)

    assert np.allclose(bicluster_project_dense(data, vectors), captured["projected"])


def test_bicluster_piecewise_projection_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_piecewise_projection import (
        bicluster_piecewise_residual_norms,
        bicluster_piecewise_vector,
        bicluster_project_dense,
        bicluster_select_best_piecewise_vectors,
    )

    with pytest.raises(Exception):
        bicluster_piecewise_vector(np.array([[1.0], [2.0]], dtype=np.float64), np.array([0, 2], dtype=np.int64))

    with pytest.raises(Exception):
        bicluster_piecewise_residual_norms(
            np.ones((2, 3), dtype=np.float64),
            np.ones((3, 3), dtype=np.float64),
        )

    with pytest.raises(Exception):
        bicluster_select_best_piecewise_vectors(
            np.ones((2, 2), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            3,
        )

    with pytest.raises(Exception):
        bicluster_project_dense(
            np.ones((2, 3), dtype=np.float64),
            np.ones((2, 2), dtype=np.float64),
        )
