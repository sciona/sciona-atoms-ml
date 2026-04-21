from __future__ import annotations

import numpy as np
import pytest
from sklearn.manifold import Isomap as SklearnIsomap


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [3.0, 2.0],
        ],
        dtype=np.float64,
    )


def test_isomap_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.manifold import (
        IsomapState,
        isomap_fit,
        isomap_geodesic_distances,
        isomap_neighbors_graph,
        isomap_reconstruction_error,
        isomap_transform,
    )

    assert IsomapState is not None
    assert callable(isomap_neighbors_graph)
    assert callable(isomap_geodesic_distances)
    assert callable(isomap_fit)
    assert callable(isomap_transform)
    assert callable(isomap_reconstruction_error)


def test_isomap_fit_matches_sklearn_dense_path() -> None:
    from sciona.atoms.ml.sklearn.manifold import isomap_fit

    X = _data()
    state = isomap_fit(X, n_neighbors=3, n_components=2, eigen_solver="dense")
    expected = SklearnIsomap(n_neighbors=3, n_components=2, eigen_solver="dense").fit(X)

    assert np.allclose(state.embedding, expected.embedding_)
    assert np.allclose(state.dist_matrix, expected.dist_matrix_)
    assert np.allclose(state.eigenvalues, expected.kernel_pca_.eigenvalues_)
    assert np.allclose(state.eigenvectors, expected.kernel_pca_.eigenvectors_)
    assert np.allclose(state.kernel_centerer_rows, expected.kernel_pca_._centerer.K_fit_rows_)
    assert np.isclose(state.kernel_centerer_all, expected.kernel_pca_._centerer.K_fit_all_)
    assert state.n_features_in == expected.n_features_in_


def test_isomap_graph_and_geodesic_distances_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.manifold import isomap_geodesic_distances, isomap_neighbors_graph

    X = _data()
    graph = isomap_neighbors_graph(X, n_neighbors=3)
    distances = isomap_geodesic_distances(graph)
    expected = SklearnIsomap(n_neighbors=3, n_components=2, eigen_solver="dense").fit(X)

    assert np.allclose(distances, expected.dist_matrix_)


def test_isomap_transform_and_reconstruction_error_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.manifold import isomap_fit, isomap_reconstruction_error, isomap_transform

    X = _data()
    query = X[[0, 3, 6]] + 0.05
    state = isomap_fit(X, n_neighbors=3, n_components=2, eigen_solver="dense")
    expected = SklearnIsomap(n_neighbors=3, n_components=2, eigen_solver="dense").fit(X)

    assert np.allclose(isomap_transform(query, state), expected.transform(query))
    assert np.isclose(isomap_reconstruction_error(state), expected.reconstruction_error())


def test_isomap_rejects_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold import isomap_fit, isomap_geodesic_distances, isomap_neighbors_graph

    X = _data()
    with pytest.raises(Exception):
        isomap_fit(X, radius=1.0, n_neighbors=3)
    with pytest.raises(Exception):
        isomap_fit(X, n_neighbors=3, eigen_solver="arpack")
    with pytest.raises(Exception):
        isomap_fit(X, n_neighbors=3, metric="precomputed")
    with pytest.raises(Exception):
        isomap_neighbors_graph(X, n_neighbors=X.shape[0])
    disconnected = np.zeros((4, 4), dtype=np.float64)
    disconnected[:2, :2] = [[0.0, 1.0], [1.0, 0.0]]
    disconnected[2:, 2:] = [[0.0, 1.0], [1.0, 0.0]]
    with pytest.raises(Exception):
        isomap_geodesic_distances(disconnected)
