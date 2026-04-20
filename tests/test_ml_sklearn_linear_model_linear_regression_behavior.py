from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.5],
            [3.0, 4.0, 1.5],
            [4.0, 2.0, 2.5],
        ],
        dtype=np.float64,
    )
    y = X @ np.array([1.25, -0.5, 2.0], dtype=np.float64) + 3.0
    return X, y


def test_linear_regression_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import LinearRegressionState, linear_regression_fit, linear_regression_predict

    assert LinearRegressionState is not None
    assert callable(linear_regression_fit)
    assert callable(linear_regression_predict)


def test_linear_regression_fit_and_predict_match_sklearn_single_output() -> None:
    from sciona.atoms.ml.sklearn.linear_model import linear_regression_fit, linear_regression_predict

    X, y = _data()
    state = linear_regression_fit(X, y)
    expected = SklearnLinearRegression().fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept[0], expected.intercept_)
    assert state.rank == expected.rank_
    assert np.allclose(state.singular, expected.singular_)
    assert state.n_features_in == expected.n_features_in_
    assert np.allclose(linear_regression_predict(X, state), expected.predict(X))


def test_linear_regression_fit_and_predict_match_sklearn_multioutput_weighted() -> None:
    from sciona.atoms.ml.sklearn.linear_model import linear_regression_fit, linear_regression_predict

    X, y_1d = _data()
    y = np.column_stack([y_1d, 2.0 * y_1d - 1.0])
    weights = np.array([1.0, 0.5, 1.5, 2.0, 0.75], dtype=np.float64)
    state = linear_regression_fit(X, y, sample_weight=weights)
    expected = SklearnLinearRegression().fit(X, y, sample_weight=weights)

    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept, expected.intercept_)
    assert np.allclose(linear_regression_predict(X[:3], state), expected.predict(X[:3]))


def test_linear_regression_fit_without_intercept_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import linear_regression_fit, linear_regression_predict

    X, y = _data()
    state = linear_regression_fit(X, y, fit_intercept=False)
    expected = SklearnLinearRegression(fit_intercept=False).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept[0], expected.intercept_)
    assert np.allclose(linear_regression_predict(X, state), expected.predict(X))


def test_linear_regression_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model import linear_regression_fit, linear_regression_predict

    X, y = _data()
    with pytest.raises(Exception):
        linear_regression_fit(X, y[:-1])
    with pytest.raises(Exception):
        linear_regression_fit(X, y, sample_weight=(1.0, 2.0))
    with pytest.raises(Exception):
        linear_regression_fit(X, y, positive=True)
    with pytest.raises(Exception):
        linear_regression_fit(X, y, tol=-1.0)

    state = linear_regression_fit(X, y)
    with pytest.raises(Exception):
        linear_regression_predict(np.ones((2, 2), dtype=np.float64), state)
