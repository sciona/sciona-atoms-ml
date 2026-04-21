from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsRegressor as SklearnKNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor as SklearnRadiusNeighborsRegressor


def _regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0.0, 1.0, 2.0, 4.0, 6.0], dtype=np.float64)
    y_multi = np.column_stack([y, y + 10.0])
    return X, y, y_multi


def test_neighbors_regressor_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        NeighborsRegressorState,
        kneighbors_regressor_fit,
        kneighbors_regressor_predict,
        radius_neighbors_regressor_fit,
        radius_neighbors_regressor_predict,
    )

    assert NeighborsRegressorState is not None
    assert callable(kneighbors_regressor_fit)
    assert callable(kneighbors_regressor_predict)
    assert callable(radius_neighbors_regressor_fit)
    assert callable(radius_neighbors_regressor_predict)


def test_kneighbors_regressor_matches_sklearn_uniform_and_distance() -> None:
    from sciona.atoms.ml.sklearn.neighbors import kneighbors_regressor_fit, kneighbors_regressor_predict

    X, y, y_multi = _regression_data()
    query = np.array([[0.5, 0.0], [3.5, 0.5], [4.0, 0.0]], dtype=np.float64)
    for target in (y, y_multi):
        for weights in ("uniform", "distance"):
            state = kneighbors_regressor_fit(X, target, n_neighbors=3, weights=weights)
            expected = SklearnKNeighborsRegressor(n_neighbors=3, weights=weights).fit(X, target)
            assert np.allclose(kneighbors_regressor_predict(query, state), expected.predict(query))


def test_kneighbors_regressor_matches_sklearn_manhattan_distance() -> None:
    from sciona.atoms.ml.sklearn.neighbors import kneighbors_regressor_fit, kneighbors_regressor_predict

    X, y, _ = _regression_data()
    query = np.array([[0.5, 0.0], [4.0, 1.0]], dtype=np.float64)
    state = kneighbors_regressor_fit(X, y, n_neighbors=2, p=1.0, weights="distance")
    expected = SklearnKNeighborsRegressor(n_neighbors=2, p=1.0, weights="distance").fit(X, y)
    assert np.allclose(kneighbors_regressor_predict(query, state), expected.predict(query))


def test_radius_neighbors_regressor_matches_sklearn_uniform_and_distance() -> None:
    from sciona.atoms.ml.sklearn.neighbors import radius_neighbors_regressor_fit, radius_neighbors_regressor_predict

    X, y, y_multi = _regression_data()
    query = np.array([[0.5, 0.0], [3.5, 0.5], [4.0, 1.0]], dtype=np.float64)
    for target in (y, y_multi):
        for weights in ("uniform", "distance"):
            state = radius_neighbors_regressor_fit(X, target, radius=1.5, weights=weights)
            expected = SklearnRadiusNeighborsRegressor(radius=1.5, weights=weights).fit(X, target)
            assert np.allclose(radius_neighbors_regressor_predict(query, state), expected.predict(query))


def test_neighbors_regressor_atoms_reject_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        kneighbors_regressor_fit,
        radius_neighbors_regressor_fit,
        radius_neighbors_regressor_predict,
    )

    X, y, _ = _regression_data()
    with pytest.raises(Exception):
        kneighbors_regressor_fit(X, y, n_neighbors=0)
    with pytest.raises(Exception):
        kneighbors_regressor_fit(X, y, weights="callable")
    with pytest.raises(Exception):
        radius_neighbors_regressor_fit(X, y, radius=-1.0)
    with pytest.raises(Exception):
        radius_neighbors_regressor_fit(X, y, metric="euclidean")
    state = radius_neighbors_regressor_fit(X, y, radius=0.25)
    with pytest.raises(Exception):
        radius_neighbors_regressor_predict(np.array([[10.0, 10.0]], dtype=np.float64), state)
