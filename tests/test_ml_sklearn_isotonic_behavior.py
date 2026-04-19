from __future__ import annotations

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.isotonic import isotonic_regression as sklearn_isotonic_regression


def test_isotonic_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.isotonic import (
        isotonic_regression,
        isotonic_regression_fit,
        isotonic_regression_predict,
        isotonic_regression_transform,
    )

    assert callable(isotonic_regression)
    assert callable(isotonic_regression_fit)
    assert callable(isotonic_regression_predict)
    assert callable(isotonic_regression_transform)


def test_isotonic_regression_matches_sklearn_increasing_and_bounds() -> None:
    from sciona.atoms.ml.sklearn.isotonic import isotonic_regression

    y = np.array([5.0, 3.0, 1.0, 2.0, 8.0, 10.0, 7.0, 9.0, 6.0, 4.0])
    weights = np.array([1.0, 2.0, 1.0, 1.0, 0.5, 1.5, 1.0, 1.0, 3.0, 1.0])
    result = isotonic_regression(y, sample_weight=weights, y_min=2.0, y_max=8.0)
    expected = sklearn_isotonic_regression(y, sample_weight=weights, y_min=2.0, y_max=8.0)
    assert np.allclose(result, expected)


def test_isotonic_regression_matches_sklearn_decreasing_float32() -> None:
    from sciona.atoms.ml.sklearn.isotonic import isotonic_regression

    y = np.array([1.0, 3.0, 2.0, 6.0, 5.0], dtype=np.float32)
    result = isotonic_regression(y, increasing=False)
    expected = sklearn_isotonic_regression(y, increasing=False)
    assert result.dtype == expected.dtype
    assert np.allclose(result, expected)


def test_fit_thresholds_and_transform_match_sklearn_with_ties() -> None:
    from sciona.atoms.ml.sklearn.isotonic import isotonic_regression_fit, isotonic_regression_transform

    X = np.array([0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0])
    weights = np.array([1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0])
    sklearn_model = IsotonicRegression(out_of_bounds="clip").fit(X, y, sample_weight=weights)
    state = isotonic_regression_fit(X, y, sample_weight=weights, out_of_bounds="clip")
    T = np.array([-1.0, 0.5, 2.5, 5.0])

    assert np.allclose(state.x_thresholds, sklearn_model.X_thresholds_)
    assert np.allclose(state.y_thresholds, sklearn_model.y_thresholds_)
    assert state.x_min == sklearn_model.X_min_
    assert state.x_max == sklearn_model.X_max_
    assert state.increasing == sklearn_model.increasing_
    assert np.allclose(isotonic_regression_transform(T, state), sklearn_model.transform(T))


def test_predict_matches_sklearn_auto_decreasing() -> None:
    from sciona.atoms.ml.sklearn.isotonic import isotonic_regression_fit, isotonic_regression_predict

    X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 8.0, 6.0, 5.0, 2.0])
    sklearn_model = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(X, y)
    state = isotonic_regression_fit(X, y, increasing="auto", out_of_bounds="clip")
    T = np.array([[0.5], [1.5], [3.5]])

    assert state.increasing is False
    assert state.increasing == sklearn_model.increasing_
    assert np.allclose(isotonic_regression_predict(T, state), sklearn_model.predict(T))


def test_transform_out_of_bounds_modes_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.isotonic import isotonic_regression_fit, isotonic_regression_transform

    X = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 1.5, 3.0])
    T = np.array([-1.0, 0.5, 4.0])

    nan_state = isotonic_regression_fit(X, y, out_of_bounds="nan")
    nan_model = IsotonicRegression(out_of_bounds="nan").fit(X, y)
    assert np.allclose(isotonic_regression_transform(T, nan_state), nan_model.transform(T), equal_nan=True)

    raise_state = isotonic_regression_fit(X, y, out_of_bounds="raise")
    raise_model = IsotonicRegression(out_of_bounds="raise").fit(X, y)
    with pytest.raises(ValueError):
        isotonic_regression_transform(T, raise_state)
    with pytest.raises(ValueError):
        raise_model.transform(T)


def test_single_threshold_constant_prediction_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.isotonic import isotonic_regression_fit, isotonic_regression_transform

    X = np.array([1.0, 1.0, 1.0])
    y = np.array([2.0, 4.0, 6.0])
    sklearn_model = IsotonicRegression(out_of_bounds="raise").fit(X, y)
    state = isotonic_regression_fit(X, y, out_of_bounds="raise")
    T = np.array([0.0, 1.0, 2.0])

    assert np.allclose(isotonic_regression_transform(T, state), sklearn_model.transform(T))


def test_fit_rejects_wrong_input_shape() -> None:
    from sciona.atoms.ml.sklearn.isotonic import isotonic_regression_fit

    X = np.ones((3, 2), dtype=np.float64)
    y = np.ones(3, dtype=np.float64)
    with pytest.raises(Exception):
        isotonic_regression_fit(X, y)
