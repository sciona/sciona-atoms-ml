from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def test_iterative_initial_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative_initial_bookkeeping import (
        iterative_clear_empty_feature_missing_mask,
        iterative_empty_feature_mask,
        iterative_filter_nonempty_matrix,
        iterative_filter_nonempty_missing_mask,
        iterative_restore_empty_feature_imputations,
    )

    assert callable(iterative_empty_feature_mask)
    assert callable(iterative_filter_nonempty_matrix)
    assert callable(iterative_filter_nonempty_missing_mask)
    assert callable(iterative_clear_empty_feature_missing_mask)
    assert callable(iterative_restore_empty_feature_imputations)


def test_iterative_empty_feature_mask_matches_fit_time_state() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative_initial_bookkeeping import iterative_empty_feature_mask

    X = np.array(
        [
            [1.0, np.nan, np.nan],
            [2.0, 5.0, np.nan],
            [3.0, 6.0, np.nan],
        ],
        dtype=np.float64,
    )
    imp = IterativeImputer()
    imp.initial_imputer_ = None
    _, _, mask_missing_values, _ = imp._initial_imputation(X, in_fit=True)

    expected = imp._is_empty_feature.copy()
    result = iterative_empty_feature_mask(np.isnan(X))

    assert np.array_equal(result, expected)
    assert np.array_equal(mask_missing_values, np.isnan(X)[:, ~expected])


def test_iterative_filter_nonempty_helpers_match_drop_empty_branch() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative_initial_bookkeeping import (
        iterative_empty_feature_mask,
        iterative_filter_nonempty_matrix,
        iterative_filter_nonempty_missing_mask,
    )

    X = np.array(
        [
            [1.0, np.nan, np.nan],
            [2.0, 5.0, np.nan],
            [3.0, 6.0, np.nan],
        ],
        dtype=np.float64,
    )
    imp = IterativeImputer(initial_strategy="constant", fill_value=-1.0, keep_empty_features=False)
    imp.initial_imputer_ = None
    Xt, X_filled, mask_missing_values, _ = imp._initial_imputation(X, in_fit=True)

    is_empty = iterative_empty_feature_mask(np.isnan(X))
    filtered_X = iterative_filter_nonempty_matrix(X, is_empty)
    filtered_mask = iterative_filter_nonempty_missing_mask(np.isnan(X), is_empty)
    filtered_filled = iterative_filter_nonempty_matrix(
        np.array([[1.0, -1.0, -1.0], [2.0, 5.0, -1.0], [3.0, 6.0, -1.0]], dtype=np.float64),
        is_empty,
    )

    assert np.array_equal(is_empty, imp._is_empty_feature)
    assert np.array_equal(filtered_X, Xt, equal_nan=True)
    assert np.array_equal(filtered_mask, mask_missing_values)
    assert np.array_equal(filtered_filled, X_filled)


def test_iterative_keep_empty_helpers_match_keep_empty_branch() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative_initial_bookkeeping import (
        iterative_clear_empty_feature_missing_mask,
        iterative_empty_feature_mask,
        iterative_restore_empty_feature_imputations,
    )

    X = np.array(
        [
            [1.0, np.nan, np.nan],
            [2.0, 5.0, np.nan],
            [3.0, 6.0, np.nan],
        ],
        dtype=np.float64,
    )
    imp = IterativeImputer(initial_strategy="mean", keep_empty_features=True)
    imp.initial_imputer_ = None
    Xt, X_filled, mask_missing_values, _ = imp._initial_imputation(X, in_fit=True)

    is_empty = iterative_empty_feature_mask(np.isnan(X))
    cleared_mask = iterative_clear_empty_feature_missing_mask(np.isnan(X), is_empty)
    restored = iterative_restore_empty_feature_imputations(X, X_filled, is_empty)

    assert np.array_equal(is_empty, imp._is_empty_feature)
    assert np.array_equal(cleared_mask, mask_missing_values)
    assert np.array_equal(restored, Xt, equal_nan=True)


def test_iterative_filter_nonempty_matrix_handles_all_empty_columns() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative_initial_bookkeeping import (
        iterative_filter_nonempty_matrix,
        iterative_filter_nonempty_missing_mask,
    )

    X = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)
    mask = np.isnan(X)
    is_empty = np.array([True, True], dtype=np.bool_)

    filtered_X = iterative_filter_nonempty_matrix(X, is_empty)
    filtered_mask = iterative_filter_nonempty_missing_mask(mask, is_empty)

    assert filtered_X.shape == (2, 0)
    assert filtered_mask.shape == (2, 0)


def test_contracts_reject_invalid_iterative_initial_bookkeeping_inputs() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative_initial_bookkeeping import (
        iterative_clear_empty_feature_missing_mask,
        iterative_empty_feature_mask,
        iterative_filter_nonempty_matrix,
        iterative_filter_nonempty_missing_mask,
        iterative_restore_empty_feature_imputations,
    )

    with pytest.raises(ViolationError):
        iterative_empty_feature_mask(np.array([[1, 0]], dtype=np.int64))

    with pytest.raises(ViolationError):
        iterative_filter_nonempty_matrix(np.ones((2, 2), dtype=np.float64), np.array([True], dtype=np.bool_))

    with pytest.raises(ViolationError):
        iterative_filter_nonempty_missing_mask(np.array([[1, 0]], dtype=np.int64), np.array([False, True], dtype=np.bool_))

    with pytest.raises(ViolationError):
        iterative_clear_empty_feature_missing_mask(np.array([[True, False]], dtype=np.bool_), np.array([False], dtype=np.bool_))

    with pytest.raises(ViolationError):
        iterative_restore_empty_feature_imputations(
            np.array([[1.0, np.nan]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            np.array([False, True], dtype=np.bool_),
        )
