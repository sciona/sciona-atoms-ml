from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import NearestCentroid as SklearnNearestCentroid


def _classification_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -2.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [3.0, 2.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    return X, y


def test_nearest_centroid_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        NearestCentroidState,
        nearest_centroid_decision_function,
        nearest_centroid_fit,
        nearest_centroid_predict,
        nearest_centroid_predict_log_proba,
        nearest_centroid_predict_proba,
    )

    assert NearestCentroidState is not None
    assert callable(nearest_centroid_fit)
    assert callable(nearest_centroid_predict)
    assert callable(nearest_centroid_decision_function)
    assert callable(nearest_centroid_predict_log_proba)
    assert callable(nearest_centroid_predict_proba)


def test_nearest_centroid_fit_predict_matches_sklearn_metrics() -> None:
    from sciona.atoms.ml.sklearn.neighbors import nearest_centroid_fit, nearest_centroid_predict

    X, y = _classification_data()
    query = np.array([[-1.5, -1.0], [1.5, 1.0], [0.0, 0.0]], dtype=np.float64)
    for metric in ("euclidean", "manhattan"):
        state = nearest_centroid_fit(X, y, metric=metric)
        expected = SklearnNearestCentroid(metric=metric).fit(X, y)
        assert np.allclose(state.centroids, expected.centroids_)
        assert np.allclose(state.deviations, expected.deviations_)
        assert np.allclose(state.within_class_std_dev, expected.within_class_std_dev_)
        assert np.allclose(nearest_centroid_predict(query, state), expected.predict(query))


def test_nearest_centroid_shrinkage_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.neighbors import nearest_centroid_fit, nearest_centroid_predict

    X, y = _classification_data()
    query = np.array([[-1.5, -1.0], [1.5, 1.0], [0.0, 0.0]], dtype=np.float64)
    state = nearest_centroid_fit(X, y, shrink_threshold=0.1)
    expected = SklearnNearestCentroid(shrink_threshold=0.1).fit(X, y)
    assert np.allclose(state.centroids, expected.centroids_)
    assert np.allclose(state.deviations, expected.deviations_)
    assert np.allclose(nearest_centroid_predict(query, state), expected.predict(query))


def test_nearest_centroid_probability_methods_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        nearest_centroid_decision_function,
        nearest_centroid_fit,
        nearest_centroid_predict,
        nearest_centroid_predict_log_proba,
        nearest_centroid_predict_proba,
    )

    X, y = _classification_data()
    query = np.array([[-1.5, -1.0], [1.5, 1.0], [0.0, 0.0]], dtype=np.float64)
    state = nearest_centroid_fit(X, y, priors="empirical")
    expected = SklearnNearestCentroid(priors="empirical").fit(X, y)
    assert np.allclose(nearest_centroid_predict(query, state), expected.predict(query))
    assert np.allclose(nearest_centroid_decision_function(query, state), expected.decision_function(query))
    assert np.allclose(nearest_centroid_predict_log_proba(query, state), expected.predict_log_proba(query))
    assert np.allclose(nearest_centroid_predict_proba(query, state), expected.predict_proba(query))


def test_nearest_centroid_atoms_reject_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        nearest_centroid_decision_function,
        nearest_centroid_fit,
        nearest_centroid_predict_proba,
    )

    X, y = _classification_data()
    with pytest.raises(Exception):
        nearest_centroid_fit(X, np.zeros_like(y))
    with pytest.raises(Exception):
        nearest_centroid_fit(X, y, metric="cosine")
    with pytest.raises(Exception):
        nearest_centroid_fit(X, y, priors=(0.5,))
    manhattan_state = nearest_centroid_fit(X, y, metric="manhattan")
    with pytest.raises(Exception):
        nearest_centroid_decision_function(X, manhattan_state)
    with pytest.raises(Exception):
        nearest_centroid_predict_proba(X, manhattan_state)
