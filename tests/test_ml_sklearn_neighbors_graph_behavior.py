from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsTransformer as SklearnKNeighborsTransformer
from sklearn.neighbors import RadiusNeighborsTransformer as SklearnRadiusNeighborsTransformer
from sklearn.neighbors import kneighbors_graph as sklearn_kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph as sklearn_radius_neighbors_graph


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )


def test_neighbors_graph_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        NeighborsGraphTransformerState,
        kneighbors_graph,
        kneighbors_transform,
        kneighbors_transformer_fit,
        radius_neighbors_graph,
        radius_neighbors_transform,
        radius_neighbors_transformer_fit,
    )

    assert NeighborsGraphTransformerState is not None
    assert callable(kneighbors_graph)
    assert callable(radius_neighbors_graph)
    assert callable(kneighbors_transformer_fit)
    assert callable(kneighbors_transform)
    assert callable(radius_neighbors_transformer_fit)
    assert callable(radius_neighbors_transform)


def test_kneighbors_graph_matches_sklearn_dense_minkowski_modes() -> None:
    from sciona.atoms.ml.sklearn.neighbors import kneighbors_graph

    X = _data()
    for mode in ("connectivity", "distance"):
        for include_self in (False, True, "auto"):
            result = kneighbors_graph(X, 2, mode=mode, include_self=include_self)
            expected = sklearn_kneighbors_graph(X, 2, mode=mode, include_self=include_self).toarray()
            assert np.allclose(result, expected)


def test_radius_neighbors_graph_matches_sklearn_dense_minkowski_modes() -> None:
    from sciona.atoms.ml.sklearn.neighbors import radius_neighbors_graph

    X = _data()
    for mode in ("connectivity", "distance"):
        for include_self in (False, True, "auto"):
            result = radius_neighbors_graph(X, 2.1, mode=mode, include_self=include_self)
            expected = sklearn_radius_neighbors_graph(X, 2.1, mode=mode, include_self=include_self).toarray()
            assert np.allclose(result, expected)


def test_kneighbors_transformer_fit_transform_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.neighbors import kneighbors_transform, kneighbors_transformer_fit

    X = _data()
    query = np.array([[0.0, 0.5], [3.0, 1.0]], dtype=np.float64)
    for mode in ("connectivity", "distance"):
        state = kneighbors_transformer_fit(X, n_neighbors=2, mode=mode)
        expected = SklearnKNeighborsTransformer(n_neighbors=2, mode=mode).fit(X)
        assert np.allclose(kneighbors_transform(X, state), expected.transform(X).toarray())
        assert np.allclose(kneighbors_transform(query, state), expected.transform(query).toarray())


def test_radius_neighbors_transformer_fit_transform_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.neighbors import radius_neighbors_transform, radius_neighbors_transformer_fit

    X = _data()
    query = np.array([[0.0, 0.5], [3.0, 1.0]], dtype=np.float64)
    for mode in ("connectivity", "distance"):
        state = radius_neighbors_transformer_fit(X, radius=2.1, mode=mode)
        expected = SklearnRadiusNeighborsTransformer(radius=2.1, mode=mode).fit(X)
        assert np.allclose(radius_neighbors_transform(X, state), expected.transform(X).toarray())
        assert np.allclose(radius_neighbors_transform(query, state), expected.transform(query).toarray())


def test_neighbors_graph_atoms_reject_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        kneighbors_graph,
        kneighbors_transformer_fit,
        radius_neighbors_graph,
        radius_neighbors_transformer_fit,
    )

    X = _data()
    with pytest.raises(Exception):
        kneighbors_graph(X, X.shape[0], include_self=False)
    with pytest.raises(Exception):
        kneighbors_graph(X, 2, metric="euclidean")
    with pytest.raises(Exception):
        radius_neighbors_graph(X, -1.0)
    with pytest.raises(Exception):
        kneighbors_transformer_fit(X, n_neighbors=2, n_jobs=2)
    with pytest.raises(Exception):
        radius_neighbors_transformer_fit(X, radius=2.1, metric_params={"w": 1.0})
