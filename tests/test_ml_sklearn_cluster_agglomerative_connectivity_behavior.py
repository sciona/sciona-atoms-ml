from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from icontract import ViolationError
from sklearn.cluster._agglomerative import _fix_connected_components, _fix_connectivity
from scipy.sparse.csgraph import connected_components


def _features() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [10.0, 0.0],
            [10.0, 3.0],
            [25.0, 0.0],
        ],
        dtype=np.float64,
    )


def _dense_connectivity() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def _distance_graph_and_labels() -> tuple[sp.lil_matrix, np.ndarray]:
    graph = sp.lil_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    return graph, np.array([0, 0, 1, 1, 2], dtype=np.int32)


def test_agglomerative_connectivity_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_connectivity import (
        agglomerative_fix_connected_components,
        agglomerative_fix_connectivity,
    )

    assert callable(agglomerative_fix_connected_components)
    assert callable(agglomerative_fix_connectivity)


def test_fix_connectivity_matches_sklearn_for_dense_disconnected_input() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_connectivity import agglomerative_fix_connectivity

    X = _features()
    connectivity = _dense_connectivity()

    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected_graph, expected_components = _fix_connectivity(X, connectivity, "euclidean")

    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual_graph, actual_components = agglomerative_fix_connectivity(X, connectivity, "euclidean")

    assert expected_components == 3
    assert actual_components == expected_components
    assert sp.isspmatrix_lil(actual_graph)
    assert np.array_equal(actual_graph.toarray(), expected_graph.toarray())
    assert connected_components(actual_graph)[0] == 1
    assert len(expected_warnings) == len(actual_warnings) == 1
    assert "connected components" in str(actual_warnings[0].message)


def test_fix_connectivity_matches_sklearn_for_sparse_input_and_symmetrizes() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_connectivity import agglomerative_fix_connectivity

    X = _features()
    connectivity = sp.csr_matrix(_dense_connectivity())

    expected_graph, expected_components = _fix_connectivity(X, connectivity, "euclidean")
    actual_graph, actual_components = agglomerative_fix_connectivity(X, connectivity, "euclidean")

    assert expected_components == actual_components == 3
    assert sp.isspmatrix_lil(actual_graph)
    assert np.array_equal(actual_graph.toarray(), expected_graph.toarray())
    assert np.array_equal(actual_graph.toarray(), actual_graph.toarray().T)


def test_fix_connected_components_matches_sklearn_for_connectivity_mode() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_connectivity import agglomerative_fix_connected_components

    X = _features()
    graph, labels = _distance_graph_and_labels()

    expected = _fix_connected_components(
        X=X,
        graph=graph.copy(),
        n_connected_components=3,
        component_labels=labels,
        metric="euclidean",
        mode="connectivity",
    )
    actual = agglomerative_fix_connected_components(
        X=X,
        graph=graph,
        n_connected_components=3,
        component_labels=labels,
        metric="euclidean",
        mode="connectivity",
    )

    assert np.array_equal(actual.toarray(), expected.toarray())
    assert connected_components(actual)[0] == 1


def test_fix_connected_components_matches_sklearn_for_precomputed_distance_mode() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_connectivity import agglomerative_fix_connected_components

    X = np.array(
        [
            [0.0, 1.0, 4.0, 7.0],
            [1.0, 0.0, 3.0, 8.0],
            [4.0, 3.0, 0.0, 2.0],
            [7.0, 8.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    graph = sp.lil_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    expected = _fix_connected_components(
        X=X,
        graph=graph.copy(),
        n_connected_components=2,
        component_labels=labels,
        metric="precomputed",
        mode="distance",
    )
    actual = agglomerative_fix_connected_components(
        X=X,
        graph=graph,
        n_connected_components=2,
        component_labels=labels,
        metric="precomputed",
        mode="distance",
    )

    assert np.array_equal(actual.toarray(), expected.toarray())
    assert actual[1, 2] == actual[2, 1] == 3.0


def test_fix_connected_components_rejects_sparse_precomputed_distance_matrix() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_connectivity import agglomerative_fix_connected_components

    X = sp.csr_matrix(np.eye(2, dtype=np.float64))
    graph = sp.lil_matrix(np.zeros((2, 2), dtype=np.float64))
    labels = np.array([0, 1], dtype=np.int32)

    with pytest.raises(RuntimeError, match="requires the full distance matrix"):
        agglomerative_fix_connected_components(
            X=X,
            graph=graph,
            n_connected_components=2,
            component_labels=labels,
            metric="precomputed",
            mode="distance",
        )


def test_contracts_reject_invalid_agglomerative_connectivity_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_connectivity import (
        agglomerative_fix_connected_components,
        agglomerative_fix_connectivity,
    )

    X = _features()
    graph, labels = _distance_graph_and_labels()

    with pytest.raises(ViolationError):
        agglomerative_fix_connectivity(X, np.ones((4, 4), dtype=np.float64), "euclidean")

    with pytest.raises(ViolationError):
        agglomerative_fix_connectivity(X[:3], np.eye(3, dtype=np.float64), "precomputed")

    with pytest.raises(ViolationError):
        agglomerative_fix_connected_components(
            X=X,
            graph=np.eye(5, dtype=np.float64),  # type: ignore[arg-type]
            n_connected_components=3,
            component_labels=labels,
        )

    with pytest.raises(ViolationError):
        agglomerative_fix_connected_components(
            X=X,
            graph=graph,
            n_connected_components=2,
            component_labels=labels,
        )

    with pytest.raises(ViolationError):
        agglomerative_fix_connected_components(
            X=X,
            graph=graph,
            n_connected_components=3,
            component_labels=np.array([0, 2, 2, 2, 2], dtype=np.int32),
        )
