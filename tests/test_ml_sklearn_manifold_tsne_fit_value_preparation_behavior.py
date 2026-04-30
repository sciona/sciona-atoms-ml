from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy.sparse import csr_matrix
from sklearn.manifold._t_sne import _joint_probabilities
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform


def test_tsne_fit_value_preparation_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_value_preparation import (
        tsne_exact_distance_matrix,
        tsne_exact_probability_vector,
        tsne_provided_layout_matrix,
        tsne_neighbor_graph_squared_data,
    )

    assert callable(tsne_exact_distance_matrix)
    assert callable(tsne_neighbor_graph_squared_data)
    assert callable(tsne_exact_probability_vector)
    assert callable(tsne_provided_layout_matrix)


def test_tsne_exact_distance_matrix_matches_sklearn_exact_postprocessing() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_value_preparation import tsne_exact_distance_matrix

    X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    euclidean = squareform(pdist(X, metric="euclidean")) ** 2
    assert np.allclose(tsne_exact_distance_matrix(euclidean, metric="euclidean"), euclidean)

    manhattan = squareform(pdist(X, metric="cityblock"))
    assert np.allclose(tsne_exact_distance_matrix(manhattan, metric="cityblock"), manhattan**2)

    with pytest.raises(ValueError, match="All distances should be positive, the metric given is not correct"):
        tsne_exact_distance_matrix(np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64), metric="precomputed")


def test_tsne_neighbor_graph_squared_data_matches_sklearn_branch() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_value_preparation import tsne_neighbor_graph_squared_data

    X = np.array([[0.0, 0.0], [1.0, 0.1], [0.2, 0.8], [0.9, 1.0]], dtype=np.float64)
    knn = NearestNeighbors(n_neighbors=2).fit(X)
    graph = knn.kneighbors_graph(mode="distance")
    expected = graph.data**2
    assert np.allclose(tsne_neighbor_graph_squared_data(graph.data.astype(np.float64)), expected)


def test_tsne_exact_probability_vector_matches_sklearn_assertion_shell() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_value_preparation import tsne_exact_probability_vector

    X = np.array([[0.0, 0.0], [1.0, 0.1], [0.2, 0.8], [0.9, 1.0]], dtype=np.float64)
    distances = squareform(pdist(X, "sqeuclidean"))
    P = _joint_probabilities(distances.copy(), 2.0, 0)
    assert np.allclose(tsne_exact_probability_vector(P), P)

    with pytest.raises(AssertionError, match="All probabilities should be non-negative"):
        tsne_exact_probability_vector(np.array([0.5, -0.1, 0.2], dtype=np.float64))

    with pytest.raises(AssertionError, match="All probabilities should be less or then equal to one"):
        tsne_exact_probability_vector(np.array([0.5, 1.1, 0.2], dtype=np.float64))


def test_tsne_provided_layout_matrix_is_passthrough() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_value_preparation import tsne_provided_layout_matrix

    init = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    assert np.allclose(tsne_provided_layout_matrix(init), init)


def test_tsne_fit_value_preparation_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_value_preparation import (
        tsne_exact_distance_matrix,
        tsne_exact_probability_vector,
        tsne_provided_layout_matrix,
        tsne_neighbor_graph_squared_data,
    )

    with pytest.raises(ViolationError):
        tsne_exact_distance_matrix(np.array([1.0, 2.0], dtype=np.float64), metric="euclidean")

    with pytest.raises(ViolationError):
        tsne_neighbor_graph_squared_data(np.array([1.0, -1.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        tsne_exact_probability_vector(np.array([[0.1, 0.2]], dtype=np.float64))

    with pytest.raises(ViolationError):
        tsne_provided_layout_matrix(np.array([0.1, 0.2], dtype=np.float64))
