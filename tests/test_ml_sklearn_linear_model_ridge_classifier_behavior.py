from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier as SklearnRidgeClassifier


def _binary_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.5],
            [3.0, 4.0, 1.5],
            [4.0, 2.0, 2.5],
            [5.0, 3.5, 0.25],
        ],
        dtype=np.float64,
    )
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    return X, y


def _multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 2.0],
            [5.0, 3.5],
            [2.5, 3.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0], dtype=np.float64)
    return X, y


def test_ridge_classifier_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        RidgeClassifierState,
        ridge_classifier_decision_function,
        ridge_classifier_fit,
        ridge_classifier_predict,
    )

    assert RidgeClassifierState is not None
    assert callable(ridge_classifier_fit)
    assert callable(ridge_classifier_decision_function)
    assert callable(ridge_classifier_predict)


def test_ridge_classifier_binary_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        ridge_classifier_decision_function,
        ridge_classifier_fit,
        ridge_classifier_predict,
    )

    X, y = _binary_data()
    state = ridge_classifier_fit(X, y, alpha=0.75, solver="cholesky")
    expected = SklearnRidgeClassifier(alpha=0.75, solver="cholesky").fit(X, y)

    assert np.allclose(state.classes, expected.classes_)
    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept, np.atleast_1d(expected.intercept_))
    assert np.allclose(ridge_classifier_decision_function(X, state), expected.decision_function(X))
    assert np.allclose(ridge_classifier_predict(X, state), expected.predict(X))


def test_ridge_classifier_multiclass_weighted_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        ridge_classifier_decision_function,
        ridge_classifier_fit,
        ridge_classifier_predict,
    )

    X, y = _multiclass_data()
    weights = np.array([1.0, 0.5, 1.5, 2.0, 0.75, 1.25, 0.8], dtype=np.float64)
    state = ridge_classifier_fit(X, y, alpha=0.5, solver="cholesky", sample_weight=weights, class_weight="balanced")
    expected = SklearnRidgeClassifier(alpha=0.5, solver="cholesky", class_weight="balanced").fit(X, y, sample_weight=weights)

    assert np.allclose(state.classes, expected.classes_)
    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept, expected.intercept_)
    assert np.allclose(ridge_classifier_decision_function(X[:4], state), expected.decision_function(X[:4]))
    assert np.allclose(ridge_classifier_predict(X[:4], state), expected.predict(X[:4]))


def test_ridge_classifier_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_classifier_fit, ridge_classifier_predict

    X, y = _binary_data()
    with pytest.raises(Exception):
        ridge_classifier_fit(X, y[:-1])
    with pytest.raises(Exception):
        ridge_classifier_fit(X, np.zeros_like(y))
    with pytest.raises(Exception):
        ridge_classifier_fit(X, y, solver="svd")
    with pytest.raises(Exception):
        ridge_classifier_fit(X, y, positive=True)
    with pytest.raises(Exception):
        ridge_classifier_fit(X, y, class_weight={0.0: -1.0})

    state = ridge_classifier_fit(X, y)
    with pytest.raises(Exception):
        ridge_classifier_predict(np.ones((2, 2), dtype=np.float64), state)
