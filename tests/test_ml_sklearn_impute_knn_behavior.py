from __future__ import annotations

import numpy as np
import pytest
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances as sklearn_nan_euclidean_distances


def _data() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.0, np.nan, np.nan],
            [3.0, 4.0, 3.0, np.nan],
            [np.nan, 6.0, 5.0, np.nan],
            [8.0, 8.0, 7.0, np.nan],
        ],
        dtype=np.float64,
    )


def test_knn_imputer_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.impute import (
        KNNImputerState,
        knn_imputer_calc_impute,
        knn_imputer_fit,
        knn_imputer_transform,
        nan_euclidean_distances,
    )

    assert KNNImputerState is not None
    assert callable(knn_imputer_calc_impute)
    assert callable(knn_imputer_fit)
    assert callable(knn_imputer_transform)
    assert callable(nan_euclidean_distances)


def test_nan_euclidean_distances_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.impute import nan_euclidean_distances

    X = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 4.0]], dtype=np.float64)
    Y = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 7.0]], dtype=np.float64)

    assert np.allclose(nan_euclidean_distances(X, Y), sklearn_nan_euclidean_distances(X, Y), equal_nan=True)


def test_knn_imputer_fit_transform_matches_sklearn_uniform() -> None:
    from sciona.atoms.ml.sklearn.impute import knn_imputer_fit, knn_imputer_transform

    X = _data()
    query = np.array([[1.0, np.nan, np.nan, np.nan], [np.nan, 7.0, 6.0, np.nan]], dtype=np.float64)
    expected = KNNImputer(n_neighbors=2).fit(X)
    state = knn_imputer_fit(X, n_neighbors=2)

    assert np.array_equal(state.mask_fit_X, np.isnan(expected._fit_X))
    assert np.array_equal(state.valid_mask, expected._valid_mask)
    assert np.allclose(knn_imputer_transform(query, state), expected.transform(query))


def test_knn_imputer_fit_transform_matches_sklearn_distance_weights() -> None:
    from sciona.atoms.ml.sklearn.impute import knn_imputer_fit, knn_imputer_transform

    X = _data()
    query = np.array([[2.0, np.nan, np.nan, np.nan], [np.nan, 7.0, 6.0, np.nan]], dtype=np.float64)
    expected = KNNImputer(n_neighbors=3, weights="distance").fit(X)
    state = knn_imputer_fit(X, n_neighbors=3, weights="distance")

    assert np.allclose(knn_imputer_transform(query, state), expected.transform(query))


def test_knn_imputer_keep_empty_features_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.impute import knn_imputer_fit, knn_imputer_transform

    X = _data()
    query = np.array([[np.nan, 7.0, 6.0, np.nan]], dtype=np.float64)
    expected = KNNImputer(n_neighbors=2, keep_empty_features=True).fit(X)
    state = knn_imputer_fit(X, n_neighbors=2, keep_empty_features=True)

    assert np.allclose(knn_imputer_transform(query, state), expected.transform(query))


def test_knn_imputer_calc_impute_matches_expected_weighted_average() -> None:
    from sciona.atoms.ml.sklearn.impute import knn_imputer_calc_impute

    distances = np.array([[1.0, 2.0, 3.0], [0.0, 2.0, 4.0]], dtype=np.float64)
    values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    mask = np.array([False, False, False], dtype=np.bool_)

    uniform = knn_imputer_calc_impute(distances, values, mask, n_neighbors=2, weights="uniform")
    distance = knn_imputer_calc_impute(distances, values, mask, n_neighbors=2, weights="distance")

    assert np.allclose(uniform, np.array([15.0, 15.0], dtype=np.float64))
    assert np.allclose(distance, np.array([13.333333333333334, 10.0], dtype=np.float64))


def test_knn_imputer_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.impute import knn_imputer_fit, knn_imputer_transform, nan_euclidean_distances

    state = knn_imputer_fit(_data(), n_neighbors=2)
    with pytest.raises(Exception):
        knn_imputer_fit(_data(), n_neighbors=0)
    with pytest.raises(Exception):
        knn_imputer_fit(_data(), weights="bad")
    with pytest.raises(Exception):
        knn_imputer_transform(np.ones((2, 2), dtype=np.float64), state)
    with pytest.raises(Exception):
        nan_euclidean_distances(np.ones((2, 2), dtype=np.float64), np.ones((2, 3), dtype=np.float64))
