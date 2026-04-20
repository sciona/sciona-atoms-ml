from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.linear_model import ridge_regression as sklearn_ridge_regression


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
    y = X @ np.array([1.25, -0.5, 2.0], dtype=np.float64) + 3.0
    return X, y


def test_ridge_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import RidgeState, ridge_fit, ridge_predict, ridge_regression

    assert RidgeState is not None
    assert callable(ridge_regression)
    assert callable(ridge_fit)
    assert callable(ridge_predict)


def test_ridge_regression_matches_sklearn_single_output() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_regression

    X, y = _data()
    result = ridge_regression(X, y, alpha=0.75, solver="cholesky")
    expected = sklearn_ridge_regression(X, y, alpha=0.75, solver="cholesky")
    assert np.allclose(result, expected)


def test_ridge_fit_and_predict_match_sklearn_single_output() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_fit, ridge_predict

    X, y = _data()
    state = ridge_fit(X, y, alpha=0.75, solver="cholesky")
    expected = SklearnRidge(alpha=0.75, solver="cholesky").fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept[0], expected.intercept_)
    assert state.n_features_in == expected.n_features_in_
    assert np.allclose(ridge_predict(X, state), expected.predict(X))


def test_ridge_fit_and_predict_match_sklearn_multioutput_weighted() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_fit, ridge_predict

    X, y_1d = _data()
    y = np.column_stack([y_1d, 2.0 * y_1d - 1.0])
    weights = np.array([1.0, 0.5, 1.5, 2.0, 0.75, 1.25], dtype=np.float64)
    alpha = np.array([0.25, 0.5], dtype=np.float64)
    state = ridge_fit(X, y, alpha=alpha, solver="cholesky", sample_weight=weights)
    expected = SklearnRidge(alpha=alpha, solver="cholesky").fit(X, y, sample_weight=weights)

    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept, expected.intercept_)
    assert np.allclose(ridge_predict(X[:3], state), expected.predict(X[:3]))


def test_ridge_without_intercept_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_fit, ridge_predict

    X, y = _data()
    state = ridge_fit(X, y, alpha=1.25, fit_intercept=False, solver="cholesky")
    expected = SklearnRidge(alpha=1.25, fit_intercept=False, solver="cholesky").fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept[0], expected.intercept_)
    assert np.allclose(ridge_predict(X, state), expected.predict(X))


def test_ridge_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_fit, ridge_predict, ridge_regression

    X, y = _data()
    with pytest.raises(Exception):
        ridge_regression(X, y[:-1])
    with pytest.raises(Exception):
        ridge_regression(X, y, alpha=-0.1)
    with pytest.raises(Exception):
        ridge_regression(X, y, solver="svd")
    with pytest.raises(Exception):
        ridge_fit(X, y, sample_weight=(1.0, 2.0))
    with pytest.raises(Exception):
        ridge_fit(X, y, positive=True)

    state = ridge_fit(X, y)
    with pytest.raises(Exception):
        ridge_predict(np.ones((2, 2), dtype=np.float64), state)
