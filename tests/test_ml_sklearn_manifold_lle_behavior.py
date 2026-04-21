from __future__ import annotations

import numpy as np
import pytest
from sklearn.manifold import LocallyLinearEmbedding as SklearnLLE
from sklearn.manifold import locally_linear_embedding as sklearn_lle
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_lle_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.manifold import (
        LocallyLinearEmbeddingState,
        lle_barycenter_graph,
        lle_barycenter_weights,
        lle_standard_reconstruction_matrix,
        locally_linear_embedding,
        locally_linear_embedding_fit,
        locally_linear_embedding_transform,
    )

    assert LocallyLinearEmbeddingState is not None
    assert callable(lle_barycenter_weights)
    assert callable(lle_barycenter_graph)
    assert callable(lle_standard_reconstruction_matrix)
    assert callable(locally_linear_embedding)
    assert callable(locally_linear_embedding_fit)
    assert callable(locally_linear_embedding_transform)


def test_lle_barycenter_graph_and_matrix_match_sklearn_standard_path() -> None:
    from sciona.atoms.ml.sklearn.manifold import lle_barycenter_graph, lle_standard_reconstruction_matrix

    X = _data()
    weights = lle_barycenter_graph(X, n_neighbors=3)
    expected_weights = barycenter_kneighbors_graph(X, n_neighbors=3).toarray()
    expected_matrix = expected_weights.T @ expected_weights - expected_weights.T - expected_weights
    expected_matrix.flat[:: expected_matrix.shape[0] + 1] += 1.0

    assert np.allclose(weights, expected_weights)
    assert np.allclose(lle_standard_reconstruction_matrix(weights), expected_matrix)


def test_locally_linear_embedding_matches_sklearn_function_dense_standard() -> None:
    from sciona.atoms.ml.sklearn.manifold import locally_linear_embedding

    X = _data()
    state = locally_linear_embedding(X, n_neighbors=3, n_components=2, eigen_solver="dense")
    expected_embedding, expected_error = sklearn_lle(X, n_neighbors=3, n_components=2, eigen_solver="dense")

    assert np.allclose(state.embedding, expected_embedding)
    assert np.isclose(state.reconstruction_error, expected_error)


def test_locally_linear_embedding_fit_and_transform_match_sklearn_estimator() -> None:
    from sciona.atoms.ml.sklearn.manifold import locally_linear_embedding_fit, locally_linear_embedding_transform

    X = _data()
    query = X[[0, 3, 7]] + 0.03
    state = locally_linear_embedding_fit(X, n_neighbors=3, n_components=2, eigen_solver="dense")
    expected = SklearnLLE(n_neighbors=3, n_components=2, eigen_solver="dense").fit(X)

    assert np.allclose(state.embedding, expected.embedding_)
    assert np.isclose(state.reconstruction_error, expected.reconstruction_error_)
    assert np.allclose(locally_linear_embedding_transform(query, state), expected.transform(query))


def test_lle_rejects_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold import lle_barycenter_graph, locally_linear_embedding, locally_linear_embedding_fit

    X = _data()
    with pytest.raises(Exception):
        locally_linear_embedding(X, n_neighbors=3, n_components=2, eigen_solver="arpack")
    with pytest.raises(Exception):
        locally_linear_embedding(X, n_neighbors=3, n_components=2, method="hessian")
    with pytest.raises(Exception):
        locally_linear_embedding(X, n_neighbors=3, n_components=4, eigen_solver="dense")
    with pytest.raises(Exception):
        locally_linear_embedding_fit(X, n_neighbors=3, n_components=2, n_jobs=2)
    with pytest.raises(Exception):
        lle_barycenter_graph(X, n_neighbors=X.shape[0])
