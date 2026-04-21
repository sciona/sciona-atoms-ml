from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import BayesianRidge as SklearnBayesianRidge


def _data() -> tuple[np.ndarray, np.ndarray]:
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
    y = X @ np.array([1.2, -0.7, 2.1], dtype=np.float64) + 0.5
    return X, y


def test_bayesian_ridge_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        BayesianRidgeState,
        bayesian_ridge_fit,
        bayesian_ridge_predict,
        bayesian_ridge_predict_std,
    )

    assert BayesianRidgeState is not None
    assert callable(bayesian_ridge_fit)
    assert callable(bayesian_ridge_predict)
    assert callable(bayesian_ridge_predict_std)


def test_bayesian_ridge_fit_predict_and_scores_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import bayesian_ridge_fit, bayesian_ridge_predict, bayesian_ridge_predict_std

    X, y = _data()
    state = bayesian_ridge_fit(X, y, compute_score=True)
    expected = SklearnBayesianRidge(compute_score=True).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.isclose(state.alpha, expected.alpha_)
    assert np.isclose(state.lambda_, expected.lambda_)
    assert np.allclose(state.sigma, expected.sigma_)
    assert np.allclose(state.scores, expected.scores_)
    assert state.n_iter == expected.n_iter_
    assert state.n_features_in == expected.n_features_in_
    assert np.allclose(bayesian_ridge_predict(X, state), expected.predict(X))
    assert np.allclose(bayesian_ridge_predict_std(X, state), expected.predict(X, return_std=True)[1])


def test_bayesian_ridge_weighted_fit_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import bayesian_ridge_fit, bayesian_ridge_predict

    X, y = _data()
    weights = np.array([1.0, 0.5, 2.0, 1.5, 0.7, 1.2], dtype=np.float64)
    state = bayesian_ridge_fit(X, y, sample_weight=weights, compute_score=True)
    expected = SklearnBayesianRidge(compute_score=True).fit(X, y, sample_weight=weights)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.isclose(state.alpha, expected.alpha_)
    assert np.isclose(state.lambda_, expected.lambda_)
    assert np.allclose(state.scores, expected.scores_)
    assert np.allclose(bayesian_ridge_predict(X[:3], state), expected.predict(X[:3]))


def test_bayesian_ridge_without_intercept_and_custom_init_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import bayesian_ridge_fit, bayesian_ridge_predict

    X, y = _data()
    state = bayesian_ridge_fit(
        X,
        y,
        fit_intercept=False,
        alpha_init=2.0,
        lambda_init=0.5,
        max_iter=50,
        tol=1e-6,
    )
    expected = SklearnBayesianRidge(
        fit_intercept=False,
        alpha_init=2.0,
        lambda_init=0.5,
        max_iter=50,
        tol=1e-6,
    ).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.allclose(bayesian_ridge_predict(X, state), expected.predict(X))


def test_bayesian_ridge_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model import bayesian_ridge_fit, bayesian_ridge_predict

    X, y = _data()
    with pytest.raises(Exception):
        bayesian_ridge_fit(X, y[:-1])
    with pytest.raises(Exception):
        bayesian_ridge_fit(X, y, tol=0.0)
    with pytest.raises(Exception):
        bayesian_ridge_fit(X, y, alpha_init=-1.0)
    with pytest.raises(Exception):
        bayesian_ridge_fit(X, y, sample_weight=(1.0, 2.0))

    state = bayesian_ridge_fit(X, y)
    with pytest.raises(Exception):
        bayesian_ridge_predict(np.ones((2, 2), dtype=np.float64), state)
