from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifierCV as SklearnRidgeClassifierCV


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 4.0],
            [6.0, 3.0],
            [7.0, 5.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0], dtype=np.float64)
    return X, y


def _binarize(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    if classes.shape[0] == 2:
        return np.where(y == classes[1], 1.0, -1.0).reshape(-1, 1)
    encoded = np.full((y.shape[0], classes.shape[0]), -1.0, dtype=np.float64)
    for class_index, class_label in enumerate(classes):
        encoded[y == class_label, class_index] = 1.0
    return encoded


def _naive_loo_scores(X: np.ndarray, y: np.ndarray, alphas: np.ndarray, *, fit_intercept: bool = True) -> np.ndarray:
    encoded_y = _binarize(y)
    scores = np.empty(alphas.shape[0], dtype=np.float64)
    indices = np.arange(X.shape[0])
    for alpha_index, alpha in enumerate(alphas):
        predictions = np.empty_like(encoded_y, dtype=np.float64)
        for held_out in range(X.shape[0]):
            train_mask = indices != held_out
            train_y = encoded_y[train_mask, 0] if encoded_y.shape[1] == 1 else encoded_y[train_mask]
            estimator = Ridge(alpha=float(alpha), fit_intercept=fit_intercept, solver="cholesky").fit(X[train_mask], train_y)
            predictions[held_out] = np.asarray(estimator.predict(X[held_out : held_out + 1])).reshape(1, -1)[0]
        scores[alpha_index] = -float(np.mean((encoded_y - predictions) ** 2))
    return scores


def test_ridge_classifier_cv_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        RidgeClassifierCVState,
        ridge_classifier_cv_decision_function,
        ridge_classifier_cv_fit,
        ridge_classifier_cv_predict,
        ridge_classifier_cv_scores,
    )

    assert RidgeClassifierCVState is not None
    assert callable(ridge_classifier_cv_scores)
    assert callable(ridge_classifier_cv_fit)
    assert callable(ridge_classifier_cv_decision_function)
    assert callable(ridge_classifier_cv_predict)


def test_ridge_classifier_cv_scores_match_leave_one_out_scores() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_classifier_cv_scores

    X, y = _data()
    alphas = np.array([0.1, 0.5, 1.0, 5.0], dtype=np.float64)
    result = ridge_classifier_cv_scores(X, y, alphas)
    expected = _naive_loo_scores(X, y, alphas)

    assert np.allclose(result, expected)


def test_ridge_classifier_cv_fit_predict_and_scores_match_sklearn_multiclass() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        ridge_classifier_cv_decision_function,
        ridge_classifier_cv_fit,
        ridge_classifier_cv_predict,
    )

    X, y = _data()
    alphas = np.array([0.1, 0.5, 1.0, 5.0], dtype=np.float64)
    state = ridge_classifier_cv_fit(X, y, alphas=alphas)
    expected = SklearnRidgeClassifierCV(alphas=alphas, cv=None, scoring=None).fit(X, y)

    assert np.allclose(state.alpha[0], expected.alpha_)
    assert np.allclose(state.best_score[0], expected.best_score_)
    assert np.allclose(state.classes, expected.classes_)
    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept, expected.intercept_)
    assert np.allclose(ridge_classifier_cv_decision_function(X[:4], state), expected.decision_function(X[:4]))
    assert np.array_equal(ridge_classifier_cv_predict(X, state), expected.predict(X))


def test_ridge_classifier_cv_fit_predict_matches_sklearn_binary_without_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_classifier_cv_decision_function, ridge_classifier_cv_fit, ridge_classifier_cv_predict

    X, _ = _data()
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    alphas = np.array([0.1, 1.0, 5.0], dtype=np.float64)
    state = ridge_classifier_cv_fit(X, y, alphas=alphas, fit_intercept=False)
    expected = SklearnRidgeClassifierCV(alphas=alphas, fit_intercept=False, cv=None, scoring=None).fit(X, y)

    assert np.allclose(state.alpha[0], expected.alpha_)
    assert np.allclose(state.best_score[0], expected.best_score_)
    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept, expected.intercept_)
    assert np.allclose(ridge_classifier_cv_decision_function(X[:3], state), expected.decision_function(X[:3]))
    assert np.array_equal(ridge_classifier_cv_predict(X, state), expected.predict(X))


def test_ridge_classifier_cv_rejects_unsupported_modes() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_classifier_cv_fit, ridge_classifier_cv_predict, ridge_classifier_cv_scores

    X, y = _data()
    state = ridge_classifier_cv_fit(X, y)
    with pytest.raises(Exception):
        ridge_classifier_cv_scores(X[:1], y[:1])
    with pytest.raises(Exception):
        ridge_classifier_cv_scores(X, np.zeros_like(y))
    with pytest.raises(Exception):
        ridge_classifier_cv_scores(X, y, alphas=(0.0, 1.0))
    with pytest.raises(Exception):
        ridge_classifier_cv_fit(X, y, scoring="accuracy")
    with pytest.raises(Exception):
        ridge_classifier_cv_fit(X, y, cv=3)
    with pytest.raises(Exception):
        ridge_classifier_cv_fit(X, y, class_weight="balanced")
    with pytest.raises(Exception):
        ridge_classifier_cv_fit(X, y, sample_weight=np.ones(X.shape[0], dtype=np.float64))
    with pytest.raises(Exception):
        ridge_classifier_cv_predict(np.ones((2, 3), dtype=np.float64), state)
