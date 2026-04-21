from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors


def _neighbor_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [3.0, 2.0],
        ],
        dtype=np.float64,
    )
    query = np.array([[0.2, 0.0], [2.0, 0.0], [3.0, 1.2]], dtype=np.float64)
    return X, query


def test_nearest_neighbors_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        NearestNeighborsState,
        nearest_neighbors_fit,
        nearest_neighbors_kneighbors,
        nearest_neighbors_kneighbors_graph,
        nearest_neighbors_radius_neighbors,
        nearest_neighbors_radius_neighbors_graph,
    )

    assert NearestNeighborsState is not None
    assert callable(nearest_neighbors_fit)
    assert callable(nearest_neighbors_kneighbors)
    assert callable(nearest_neighbors_radius_neighbors)
    assert callable(nearest_neighbors_kneighbors_graph)
    assert callable(nearest_neighbors_radius_neighbors_graph)


def test_nearest_neighbors_kneighbors_matches_sklearn_default_distance() -> None:
    from sciona.atoms.ml.sklearn.neighbors import nearest_neighbors_fit, nearest_neighbors_kneighbors

    X, query = _neighbor_data()
    state = nearest_neighbors_fit(X, n_neighbors=2)
    distances, indices = nearest_neighbors_kneighbors(query, state)
    expected_distances, expected_indices = SklearnNearestNeighbors(n_neighbors=2).fit(X).kneighbors(query)
    assert np.allclose(distances, expected_distances)
    assert np.array_equal(indices, expected_indices)


def test_nearest_neighbors_queries_match_sklearn_manhattan_override() -> None:
    from sciona.atoms.ml.sklearn.neighbors import nearest_neighbors_fit, nearest_neighbors_kneighbors

    X, query = _neighbor_data()
    state = nearest_neighbors_fit(X, n_neighbors=3, p=1.0)
    distances, indices = nearest_neighbors_kneighbors(query, state, n_neighbors=2)
    expected = SklearnNearestNeighbors(n_neighbors=3, p=1.0).fit(X)
    expected_distances, expected_indices = expected.kneighbors(query, n_neighbors=2)
    assert np.allclose(distances, expected_distances)
    assert np.array_equal(indices, expected_indices)


def test_nearest_neighbors_radius_neighbors_matches_sklearn_sorted() -> None:
    from sciona.atoms.ml.sklearn.neighbors import nearest_neighbors_fit, nearest_neighbors_radius_neighbors

    X, query = _neighbor_data()
    state = nearest_neighbors_fit(X, n_neighbors=2, radius=1.35)
    distances, indices = nearest_neighbors_radius_neighbors(query, state, sort_results=True)
    expected_distances, expected_indices = SklearnNearestNeighbors(radius=1.35).fit(X).radius_neighbors(
        query,
        sort_results=True,
    )
    assert distances.shape == expected_distances.shape
    assert indices.shape == expected_indices.shape
    for actual, expected in zip(distances, expected_distances):
        assert np.allclose(actual, expected)
    for actual, expected in zip(indices, expected_indices):
        assert np.array_equal(actual, expected)


def test_nearest_neighbors_graphs_match_sklearn_dense_views() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        nearest_neighbors_fit,
        nearest_neighbors_kneighbors_graph,
        nearest_neighbors_radius_neighbors_graph,
    )

    X, query = _neighbor_data()
    state = nearest_neighbors_fit(X, n_neighbors=2, radius=1.35)
    expected = SklearnNearestNeighbors(n_neighbors=2, radius=1.35).fit(X)
    assert np.allclose(
        nearest_neighbors_kneighbors_graph(query, state, mode="distance"),
        expected.kneighbors_graph(query, mode="distance").toarray(),
    )
    assert np.allclose(
        nearest_neighbors_radius_neighbors_graph(query, state, mode="distance", sort_results=True),
        expected.radius_neighbors_graph(query, mode="distance", sort_results=True).toarray(),
    )


def test_nearest_neighbors_rejects_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.neighbors import nearest_neighbors_fit, nearest_neighbors_kneighbors

    X, query = _neighbor_data()
    with pytest.raises(Exception):
        nearest_neighbors_fit(X, n_neighbors=0)
    with pytest.raises(Exception):
        nearest_neighbors_fit(X, radius=-1.0)
    with pytest.raises(Exception):
        nearest_neighbors_fit(X, metric="cosine")
    state = nearest_neighbors_fit(X, n_neighbors=2)
    with pytest.raises(Exception):
        nearest_neighbors_kneighbors(query, state, n_neighbors=10)
