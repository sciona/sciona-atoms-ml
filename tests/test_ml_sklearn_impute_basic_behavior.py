from __future__ import annotations

import numpy as np
import pytest
from sklearn.impute import MissingIndicator, SimpleImputer


def _data() -> np.ndarray:
    return np.array(
        [
            [1.0, np.nan, 3.0, np.nan],
            [2.0, 5.0, np.nan, np.nan],
            [2.0, 7.0, 9.0, np.nan],
        ],
        dtype=np.float64,
    )


def test_impute_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.impute import (
        MissingIndicatorState,
        SimpleImputerState,
        missing_indicator_fit,
        missing_indicator_transform,
        simple_imputer_fit,
        simple_imputer_transform,
    )

    assert SimpleImputerState is not None
    assert MissingIndicatorState is not None
    assert callable(missing_indicator_fit)
    assert callable(missing_indicator_transform)
    assert callable(simple_imputer_fit)
    assert callable(simple_imputer_transform)


def test_simple_imputer_fit_transform_matches_sklearn_mean_and_median() -> None:
    from sciona.atoms.ml.sklearn.impute import simple_imputer_fit, simple_imputer_transform

    X = _data()
    for strategy in ("mean", "median"):
        expected = SimpleImputer(strategy=strategy).fit(X)
        state = simple_imputer_fit(X, strategy=strategy)

        assert np.allclose(state.statistics, expected.statistics_, equal_nan=True)
        assert np.array_equal(simple_imputer_transform(X, state), expected.transform(X))


def test_simple_imputer_most_frequent_and_constant_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.impute import simple_imputer_fit, simple_imputer_transform

    X = _data()
    frequent = SimpleImputer(strategy="most_frequent").fit(X)
    frequent_state = simple_imputer_fit(X, strategy="most_frequent")
    constant = SimpleImputer(strategy="constant", fill_value=-1.0).fit(X)
    constant_state = simple_imputer_fit(X, strategy="constant", fill_value=-1.0)

    assert np.allclose(frequent_state.statistics, frequent.statistics_, equal_nan=True)
    assert np.array_equal(simple_imputer_transform(X, frequent_state), frequent.transform(X))
    assert np.allclose(constant_state.statistics, constant.statistics_.astype(np.float64), equal_nan=True)
    assert np.array_equal(simple_imputer_transform(X, constant_state), constant.transform(X))


def test_simple_imputer_keep_empty_features_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.impute import simple_imputer_fit, simple_imputer_transform

    X = _data()
    expected = SimpleImputer(strategy="mean", keep_empty_features=True).fit(X)
    state = simple_imputer_fit(X, strategy="mean", keep_empty_features=True)

    assert np.allclose(state.statistics, expected.statistics_, equal_nan=True)
    assert np.array_equal(simple_imputer_transform(X, state), expected.transform(X))


def test_missing_indicator_fit_transform_matches_sklearn_modes() -> None:
    from sciona.atoms.ml.sklearn.impute import missing_indicator_fit, missing_indicator_transform

    X = _data()
    query = np.array([[np.nan, 1.0, 2.0, np.nan], [3.0, np.nan, np.nan, np.nan]], dtype=np.float64)
    for features in ("missing-only", "all"):
        expected = MissingIndicator(features=features, error_on_new=False).fit(X)
        state = missing_indicator_fit(X, features=features)

        assert np.array_equal(state.features, expected.features_)
        assert np.array_equal(missing_indicator_transform(query, state, error_on_new=False), expected.transform(query))


def test_missing_indicator_rejects_new_missing_features_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.impute import missing_indicator_fit, missing_indicator_transform

    X = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64)
    query = np.array([[np.nan, np.nan]], dtype=np.float64)
    state = missing_indicator_fit(X)

    with pytest.raises(ValueError):
        missing_indicator_transform(query, state)
    with pytest.raises(ValueError):
        MissingIndicator().fit(X).transform(query)


def test_impute_atoms_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.impute import missing_indicator_fit, simple_imputer_fit, simple_imputer_transform

    X = _data()
    state = simple_imputer_fit(X)
    with pytest.raises(Exception):
        simple_imputer_fit(np.array([1.0, np.nan], dtype=np.float64))
    with pytest.raises(Exception):
        simple_imputer_fit(X, strategy="unsupported")
    with pytest.raises(Exception):
        simple_imputer_transform(np.ones((2, 2), dtype=np.float64), state)
    with pytest.raises(Exception):
        missing_indicator_fit(X, features="unsupported")
