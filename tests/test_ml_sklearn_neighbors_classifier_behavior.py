from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier as SklearnRadiusNeighborsClassifier


def _classification_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
            [5.0, 2.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    return X, y


def test_neighbors_classifier_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        NeighborsClassifierState,
        kneighbors_classifier_fit,
        kneighbors_classifier_predict,
        kneighbors_classifier_predict_proba,
        radius_neighbors_classifier_fit,
        radius_neighbors_classifier_predict,
        radius_neighbors_classifier_predict_proba,
    )

    assert NeighborsClassifierState is not None
    assert callable(kneighbors_classifier_fit)
    assert callable(kneighbors_classifier_predict)
    assert callable(kneighbors_classifier_predict_proba)
    assert callable(radius_neighbors_classifier_fit)
    assert callable(radius_neighbors_classifier_predict)
    assert callable(radius_neighbors_classifier_predict_proba)


def test_kneighbors_classifier_matches_sklearn_uniform_and_distance() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        kneighbors_classifier_fit,
        kneighbors_classifier_predict,
        kneighbors_classifier_predict_proba,
    )

    X, y = _classification_data()
    query = np.array([[0.5, 0.0], [3.5, 0.5], [4.5, 2.0]], dtype=np.float64)
    for weights in ("uniform", "distance"):
        state = kneighbors_classifier_fit(X, y, n_neighbors=3, weights=weights)
        expected = SklearnKNeighborsClassifier(n_neighbors=3, weights=weights).fit(X, y)
        assert np.allclose(kneighbors_classifier_predict(query, state), expected.predict(query))
        assert np.allclose(kneighbors_classifier_predict_proba(query, state), expected.predict_proba(query))


def test_kneighbors_classifier_matches_sklearn_manhattan_distance() -> None:
    from sciona.atoms.ml.sklearn.neighbors import kneighbors_classifier_fit, kneighbors_classifier_predict_proba

    X, y = _classification_data()
    query = np.array([[0.5, 0.0], [4.0, 1.0]], dtype=np.float64)
    state = kneighbors_classifier_fit(X, y, n_neighbors=3, p=1.0, weights="distance")
    expected = SklearnKNeighborsClassifier(n_neighbors=3, p=1.0, weights="distance").fit(X, y)
    assert np.allclose(kneighbors_classifier_predict_proba(query, state), expected.predict_proba(query))


def test_radius_neighbors_classifier_matches_sklearn_uniform_and_distance() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        radius_neighbors_classifier_fit,
        radius_neighbors_classifier_predict,
        radius_neighbors_classifier_predict_proba,
    )

    X, y = _classification_data()
    query = np.array([[0.5, 0.0], [3.5, 0.5], [4.5, 2.0]], dtype=np.float64)
    for weights in ("uniform", "distance"):
        state = radius_neighbors_classifier_fit(X, y, radius=1.5, weights=weights)
        expected = SklearnRadiusNeighborsClassifier(radius=1.5, weights=weights).fit(X, y)
        assert np.allclose(radius_neighbors_classifier_predict(query, state), expected.predict(query))
        assert np.allclose(radius_neighbors_classifier_predict_proba(query, state), expected.predict_proba(query))


def test_neighbors_classifier_atoms_reject_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        kneighbors_classifier_fit,
        radius_neighbors_classifier_fit,
        radius_neighbors_classifier_predict,
    )

    X, y = _classification_data()
    with pytest.raises(Exception):
        kneighbors_classifier_fit(X, y, n_neighbors=0)
    with pytest.raises(Exception):
        kneighbors_classifier_fit(X, np.zeros_like(y))
    with pytest.raises(Exception):
        kneighbors_classifier_fit(X, y, weights="callable")
    with pytest.raises(Exception):
        radius_neighbors_classifier_fit(X, y, outlier_label=0.0)
    state = radius_neighbors_classifier_fit(X, y, radius=0.25)
    with pytest.raises(Exception):
        radius_neighbors_classifier_predict(np.array([[10.0, 10.0]], dtype=np.float64), state)
