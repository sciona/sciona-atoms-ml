from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LassoLars as SklearnLassoLars
from sklearn.linear_model import LassoLarsIC as SklearnLassoLarsIC
from sklearn.linear_model import lars_path as sklearn_lars_path


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-2.0, 2.0, 0.5],
            [-1.0, 1.0, 1.5],
            [0.0, 0.0, -0.5],
            [1.0, 1.0, 0.25],
            [2.0, 2.0, -1.0],
            [3.0, 0.5, 2.0],
            [4.0, -0.5, 1.0],
            [5.0, 1.5, -1.5],
        ],
        dtype=np.float64,
    )
    y = (
        X @ np.array([0.4, -1.3, 0.8], dtype=np.float64)
        + 0.25
        + np.array([0.2, -0.1, 0.05, -0.2, 0.1, -0.05, 0.15, -0.15], dtype=np.float64)
    )
    return X, y


def test_lasso_lars_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        LassoLarsICState,
        LassoLarsState,
        lasso_lars_fit,
        lasso_lars_ic_fit,
        lasso_lars_ic_predict,
        lasso_lars_path,
        lasso_lars_predict,
    )

    assert LassoLarsState is not None
    assert LassoLarsICState is not None
    assert callable(lasso_lars_path)
    assert callable(lasso_lars_fit)
    assert callable(lasso_lars_predict)
    assert callable(lasso_lars_ic_fit)
    assert callable(lasso_lars_ic_predict)


def test_lasso_lars_path_matches_sklearn_lasso_method() -> None:
    from sciona.atoms.ml.sklearn.linear_model import lasso_lars_path

    X, y = _data()
    state = lasso_lars_path(X, y, max_iter=5, alpha_min=0.0)
    alphas, active, coefs = sklearn_lars_path(X, y, method="lasso", max_iter=5, alpha_min=0.0)

    assert np.allclose(state.alphas, alphas)
    assert np.array_equal(state.active, np.asarray(active, dtype=np.int64))
    assert np.allclose(state.coefs, coefs)
    assert state.method == "lasso"


def test_lasso_lars_fit_predict_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import lasso_lars_fit, lasso_lars_predict

    X, y = _data()
    state = lasso_lars_fit(X, y, alpha=0.01, max_iter=5)
    expected = SklearnLassoLars(alpha=0.01, max_iter=5).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.allclose(state.alphas, expected.alphas_)
    assert np.array_equal(state.active, np.asarray(expected.active_, dtype=np.int64))
    assert np.allclose(state.coef_path, expected.coef_path_)
    assert state.n_iter == expected.n_iter_
    assert np.allclose(lasso_lars_predict(X, state), expected.predict(X))


def test_lasso_lars_without_intercept_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import lasso_lars_fit, lasso_lars_predict

    X, y = _data()
    state = lasso_lars_fit(X, y, alpha=0.05, fit_intercept=False, max_iter=4)
    expected = SklearnLassoLars(alpha=0.05, fit_intercept=False, max_iter=4).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.allclose(lasso_lars_predict(X, state), expected.predict(X))


def test_lasso_lars_ic_matches_sklearn_bic() -> None:
    from sciona.atoms.ml.sklearn.linear_model import lasso_lars_ic_fit, lasso_lars_ic_predict

    X, y = _data()
    state = lasso_lars_ic_fit(X, y, criterion="bic", max_iter=5)
    expected = SklearnLassoLarsIC(criterion="bic", max_iter=5).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.isclose(state.alpha, expected.alpha_)
    assert np.allclose(state.alphas, expected.alphas_)
    assert np.allclose(state.criterion_values, expected.criterion_)
    assert np.isclose(state.noise_variance, expected.noise_variance_)
    assert np.allclose(lasso_lars_ic_predict(X, state), expected.predict(X))


def test_lasso_lars_rejects_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model import lasso_lars_fit, lasso_lars_ic_fit, lasso_lars_path, lasso_lars_predict

    X, y = _data()
    with pytest.raises(Exception):
        lasso_lars_path(X, y, positive=True)
    with pytest.raises(Exception):
        lasso_lars_path(X, y, return_path=False)
    with pytest.raises(Exception):
        lasso_lars_fit(X, y, fit_path=False)
    with pytest.raises(Exception):
        lasso_lars_fit(X, y, jitter=1e-4)
    with pytest.raises(Exception):
        lasso_lars_ic_fit(X, y, criterion="bad")
    with pytest.raises(Exception):
        lasso_lars_ic_fit(X[:4], y[:4])

    state = lasso_lars_fit(X, y)
    with pytest.raises(Exception):
        lasso_lars_predict(np.ones((2, 2), dtype=np.float64), state)
