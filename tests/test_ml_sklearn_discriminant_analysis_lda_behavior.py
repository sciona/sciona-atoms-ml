from __future__ import annotations

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA


def _data() -> tuple[np.ndarray, np.ndarray]:
    class_zero = np.array(
        [
            [0.0, 0.0, 0.2],
            [0.2, 0.4, 0.1],
            [0.5, -0.1, 0.4],
            [0.8, 0.3, 0.5],
            [1.0, 0.7, 0.8],
            [1.2, 0.2, 0.6],
        ],
        dtype=np.float64,
    )
    class_one = np.array(
        [
            [3.0, 3.5, 2.8],
            [3.4, 4.1, 3.3],
            [3.8, 3.2, 3.7],
            [4.2, 3.9, 4.1],
            [4.5, 4.4, 4.6],
            [4.0, 4.7, 4.2],
        ],
        dtype=np.float64,
    )
    class_two = np.array(
        [
            [-3.0, 2.5, -2.2],
            [-2.7, 2.9, -2.0],
            [-2.3, 2.2, -1.8],
            [-1.9, 2.8, -1.4],
            [-1.6, 3.3, -1.1],
            [-2.2, 3.6, -1.5],
        ],
        dtype=np.float64,
    )
    X = np.vstack([class_zero, class_one, class_two])
    y = np.repeat(np.array([0.0, 1.0, 2.0], dtype=np.float64), 6)
    return X, y


def _binary_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = _data()
    mask = y < 2.0
    return X[mask], y[mask]


def test_lda_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.discriminant_analysis import (
        LDAState,
        lda_decision_function,
        lda_fit,
        lda_predict,
        lda_predict_log_proba,
        lda_predict_proba,
        lda_transform,
    )

    assert LDAState is not None
    assert callable(lda_fit)
    assert callable(lda_decision_function)
    assert callable(lda_predict_log_proba)
    assert callable(lda_predict_proba)
    assert callable(lda_predict)
    assert callable(lda_transform)


def test_lda_fit_and_predictions_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.discriminant_analysis import (
        lda_decision_function,
        lda_fit,
        lda_predict,
        lda_predict_log_proba,
        lda_predict_proba,
        lda_transform,
    )

    X, y = _data()
    state = lda_fit(X, y, store_covariance=True)
    expected = SklearnLDA(solver="svd", store_covariance=True).fit(X, y)

    assert np.allclose(state.classes, expected.classes_)
    assert np.allclose(state.priors, expected.priors_)
    assert np.allclose(state.means, expected.means_)
    assert np.allclose(state.xbar, expected.xbar_)
    assert np.allclose(np.abs(state.scalings), np.abs(expected.scalings_))
    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept, expected.intercept_)
    assert state.covariance is not None
    assert np.allclose(state.covariance, expected.covariance_)
    assert np.allclose(state.explained_variance_ratio, expected.explained_variance_ratio_)

    query = X[[0, 5, 8, 13]]
    assert np.allclose(lda_decision_function(query, state), expected.decision_function(query))
    assert np.allclose(lda_predict_log_proba(query, state), expected.predict_log_proba(query))
    assert np.allclose(lda_predict_proba(query, state), expected.predict_proba(query))
    assert np.allclose(lda_predict(query, state), expected.predict(query))
    assert np.allclose(np.abs(lda_transform(query, state)), np.abs(expected.transform(query)))


def test_lda_binary_scores_and_probabilities_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.discriminant_analysis import (
        lda_decision_function,
        lda_fit,
        lda_predict_log_proba,
        lda_predict_proba,
    )

    X, y = _binary_data()
    state = lda_fit(X, y)
    expected = SklearnLDA(solver="svd").fit(X, y)
    query = X[[0, 4, 7, 10]]

    decision = lda_decision_function(query, state)
    assert decision.ndim == 1
    assert np.allclose(decision, expected.decision_function(query))
    assert np.allclose(lda_predict_log_proba(query, state), expected.predict_log_proba(query))
    assert np.allclose(lda_predict_proba(query, state), expected.predict_proba(query))


def test_lda_fit_with_priors_and_components_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.discriminant_analysis import lda_fit, lda_transform

    X, y = _data()
    priors = (0.2, 0.5, 0.3)
    state = lda_fit(X, y, priors=priors, n_components=1)
    expected = SklearnLDA(solver="svd", priors=np.asarray(priors), n_components=1).fit(X, y)

    assert state.covariance is None
    assert state.n_components == 1
    assert np.allclose(state.priors, expected.priors_)
    assert np.allclose(np.abs(lda_transform(X[:5], state)), np.abs(expected.transform(X[:5])))


def test_lda_rejects_prior_and_component_errors() -> None:
    from sciona.atoms.ml.sklearn.discriminant_analysis import lda_fit

    X, y = _data()
    with pytest.raises(Exception):
        lda_fit(X, y, priors=(0.5, 0.5))

    with pytest.raises(Exception):
        lda_fit(X, y, n_components=0)

    with pytest.raises(Exception):
        lda_fit(X, y, n_components=3)
