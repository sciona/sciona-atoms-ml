from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import TheilSenRegressor as SklearnTheilSenRegressor


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.5],
            [3.0, 4.0, 1.5],
            [4.0, 2.0, 2.5],
            [5.0, 3.5, 0.25],
            [6.0, 1.5, 1.25],
        ],
        dtype=np.float64,
    )
    y = X @ np.array([1.2, -0.7, 2.1], dtype=np.float64) + 0.5
    y = y.copy()
    y[-1] += 8.0
    return X, y


def test_theil_sen_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        TheilSenRegressorState,
        theil_sen_regressor_fit,
        theil_sen_regressor_predict,
    )

    assert TheilSenRegressorState is not None
    assert callable(theil_sen_regressor_fit)
    assert callable(theil_sen_regressor_predict)


def test_theil_sen_fit_predict_matches_sklearn_all_combinations() -> None:
    from sciona.atoms.ml.sklearn.linear_model import theil_sen_regressor_fit, theil_sen_regressor_predict

    X, y = _data()
    state = theil_sen_regressor_fit(X, y, random_state=0, n_jobs=1)
    expected = SklearnTheilSenRegressor(random_state=0, n_jobs=1).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.isclose(state.breakdown, expected.breakdown_)
    assert state.n_iter == expected.n_iter_
    assert state.n_subpopulation == expected.n_subpopulation_
    assert state.n_features_in == expected.n_features_in_
    assert np.allclose(theil_sen_regressor_predict(X, state), expected.predict(X))


def test_theil_sen_random_subpopulation_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import theil_sen_regressor_fit, theil_sen_regressor_predict

    X, y = _data()
    state = theil_sen_regressor_fit(X, y, max_subpopulation=4, random_state=13, n_jobs=1)
    expected = SklearnTheilSenRegressor(max_subpopulation=4, random_state=13, n_jobs=1).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.isclose(state.breakdown, expected.breakdown_)
    assert state.n_iter == expected.n_iter_
    assert state.n_subpopulation == expected.n_subpopulation_
    assert np.allclose(theil_sen_regressor_predict(X[:3], state), expected.predict(X[:3]))


def test_theil_sen_without_intercept_and_custom_subsamples_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import theil_sen_regressor_fit, theil_sen_regressor_predict

    X, y = _data()
    state = theil_sen_regressor_fit(
        X,
        y,
        fit_intercept=False,
        n_subsamples=3,
        max_iter=100,
        tol=1e-4,
        random_state=0,
        n_jobs=1,
    )
    expected = SklearnTheilSenRegressor(
        fit_intercept=False,
        n_subsamples=3,
        max_iter=100,
        tol=1e-4,
        random_state=0,
        n_jobs=1,
    ).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.isclose(state.breakdown, expected.breakdown_)
    assert np.allclose(theil_sen_regressor_predict(X, state), expected.predict(X))


def test_theil_sen_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model import theil_sen_regressor_fit, theil_sen_regressor_predict

    X, y = _data()
    with pytest.raises(Exception):
        theil_sen_regressor_fit(X, y[:-1])
    with pytest.raises(Exception):
        theil_sen_regressor_fit(X, y, tol=0.0)
    with pytest.raises(Exception):
        theil_sen_regressor_fit(X, y, max_subpopulation=0)
    with pytest.raises(Exception):
        theil_sen_regressor_fit(X, y, n_subsamples=2)
    with pytest.raises(Exception):
        theil_sen_regressor_fit(X, y, n_jobs=2)

    state = theil_sen_regressor_fit(X, y, random_state=0)
    with pytest.raises(Exception):
        theil_sen_regressor_predict(np.ones((2, 2), dtype=np.float64), state)
