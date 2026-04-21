from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import ARDRegression as SklearnARDRegression


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0, 2.0, 0.5],
            [1.0, 2.0, 3.0, 1.0],
            [2.0, 1.0, 0.5, 1.5],
            [3.0, 4.0, 1.5, 2.0],
            [4.0, 2.0, 2.5, 2.5],
            [5.0, 3.5, 0.25, 3.0],
            [6.0, 1.5, 1.25, 3.5],
        ],
        dtype=np.float64,
    )
    y = X @ np.array([1.2, -0.7, 2.1, 0.0], dtype=np.float64) + 0.5
    return X, y


def test_ard_regression_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        ARDRegressionState,
        ard_regression_fit,
        ard_regression_predict,
        ard_regression_predict_std,
    )

    assert ARDRegressionState is not None
    assert callable(ard_regression_fit)
    assert callable(ard_regression_predict)
    assert callable(ard_regression_predict_std)


def test_ard_regression_fit_predict_and_scores_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ard_regression_fit, ard_regression_predict, ard_regression_predict_std

    X, y = _data()
    state = ard_regression_fit(X, y, compute_score=True)
    expected = SklearnARDRegression(compute_score=True).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.isclose(state.alpha, expected.alpha_)
    assert np.allclose(state.lambda_, expected.lambda_)
    assert np.allclose(state.sigma, expected.sigma_)
    assert np.allclose(state.scores, expected.scores_)
    assert state.n_iter == expected.n_iter_
    assert state.n_features_in == expected.n_features_in_
    assert np.allclose(ard_regression_predict(X, state), expected.predict(X))
    assert np.allclose(ard_regression_predict_std(X, state), expected.predict(X, return_std=True)[1])


def test_ard_regression_without_intercept_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ard_regression_fit, ard_regression_predict

    X, y = _data()
    state = ard_regression_fit(X, y, fit_intercept=False, max_iter=80, tol=1e-6)
    expected = SklearnARDRegression(fit_intercept=False, max_iter=80, tol=1e-6).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.isclose(state.alpha, expected.alpha_)
    assert np.allclose(state.lambda_, expected.lambda_)
    assert np.allclose(ard_regression_predict(X, state), expected.predict(X))


def test_ard_regression_pruning_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ard_regression_fit, ard_regression_predict_std

    X, y = _data()
    state = ard_regression_fit(X, y, threshold_lambda=25.0)
    expected = SklearnARDRegression(threshold_lambda=25.0).fit(X, y)

    assert np.array_equal(state.lambda_ < state.threshold_lambda, expected.lambda_ < expected.threshold_lambda)
    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.sigma, expected.sigma_)
    assert np.allclose(ard_regression_predict_std(X[:3], state), expected.predict(X[:3], return_std=True)[1])


def test_ard_regression_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ard_regression_fit, ard_regression_predict

    X, y = _data()
    with pytest.raises(Exception):
        ard_regression_fit(X, y[:-1])
    with pytest.raises(Exception):
        ard_regression_fit(X, y, tol=0.0)
    with pytest.raises(Exception):
        ard_regression_fit(X, y, threshold_lambda=0.0)
    with pytest.raises(Exception):
        ard_regression_fit(X, y, verbose=True)

    state = ard_regression_fit(X, y)
    with pytest.raises(Exception):
        ard_regression_predict(np.ones((2, 2), dtype=np.float64), state)
