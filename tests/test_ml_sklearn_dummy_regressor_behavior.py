from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor as SklearnDummyRegressor


def test_dummy_regressor_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.dummy import DummyRegressorState, dummy_regressor_fit, dummy_regressor_predict

    assert DummyRegressorState is not None
    assert callable(dummy_regressor_fit)
    assert callable(dummy_regressor_predict)


def test_dummy_regressor_mean_predict_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.dummy import dummy_regressor_fit, dummy_regressor_predict

    X = np.arange(6, dtype=np.float64).reshape(-1, 1)
    y = np.array([1.0, 2.0, 10.0, 20.0, 100.0, -5.0], dtype=np.float64)
    state = dummy_regressor_fit(y, strategy="mean")
    expected = SklearnDummyRegressor(strategy="mean").fit(X, y)

    assert np.allclose(state.constant, expected.constant_)
    assert state.n_outputs == expected.n_outputs_
    assert np.allclose(dummy_regressor_predict(X[:3], state), expected.predict(X[:3]))


def test_dummy_regressor_median_and_quantile_match_sklearn_multioutput() -> None:
    from sciona.atoms.ml.sklearn.dummy import dummy_regressor_fit, dummy_regressor_predict

    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0], [7.0, 14.0], [9.0, 18.0], [11.0, 22.0]], dtype=np.float64)

    median_state = dummy_regressor_fit(y, strategy="median")
    median_expected = SklearnDummyRegressor(strategy="median").fit(X, y)
    assert np.allclose(median_state.constant, median_expected.constant_)
    assert np.allclose(dummy_regressor_predict(X[:2], median_state), median_expected.predict(X[:2]))

    quantile_state = dummy_regressor_fit(y, strategy="quantile", quantile=0.25)
    quantile_expected = SklearnDummyRegressor(strategy="quantile", quantile=0.25).fit(X, y)
    assert np.allclose(quantile_state.constant, quantile_expected.constant_)
    assert np.allclose(dummy_regressor_predict(X[:2], quantile_state), quantile_expected.predict(X[:2]))


def test_dummy_regressor_constant_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.dummy import dummy_regressor_fit, dummy_regressor_predict

    X = np.arange(8, dtype=np.float64).reshape(4, 2)
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    state = dummy_regressor_fit(y, strategy="constant", constant=(4.5, 9.5))
    expected = SklearnDummyRegressor(strategy="constant", constant=np.array([4.5, 9.5])).fit(X, y)

    assert np.allclose(state.constant, expected.constant_)
    assert np.allclose(dummy_regressor_predict(X[:2], state), expected.predict(X[:2]))


def test_dummy_regressor_rejects_invalid_strategy_inputs() -> None:
    from sciona.atoms.ml.sklearn.dummy import dummy_regressor_fit

    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(Exception):
        dummy_regressor_fit(y, strategy="quantile")
    with pytest.raises(Exception):
        dummy_regressor_fit(y, strategy="constant")
    with pytest.raises(Exception):
        dummy_regressor_fit(y, strategy="unknown")
