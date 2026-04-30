from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from sciona.atoms.ml.sklearn.impute.iterative import (
    iterative_limit_vector,
    iterative_ordered_feature_indices,
)
from sciona.atoms.ml.sklearn.impute.iterative_loop_bookkeeping import (
    iterative_fit_initial_return_required,
    iterative_imputations_per_round,
    iterative_missing_feature_count,
    iterative_normalized_tolerance,
    iterative_require_strict_limits,
    iterative_single_feature_return_required,
    iterative_transform_initial_return_required,
)


def _mask() -> np.ndarray:
    return np.array(
        [
            [False, True, False],
            [False, False, False],
            [False, True, True],
        ],
        dtype=np.bool_,
    )


def test_iterative_loop_bookkeeping_atoms_import() -> None:
    assert callable(iterative_fit_initial_return_required)
    assert callable(iterative_transform_initial_return_required)
    assert callable(iterative_single_feature_return_required)
    assert callable(iterative_require_strict_limits)
    assert callable(iterative_missing_feature_count)
    assert callable(iterative_normalized_tolerance)
    assert callable(iterative_imputations_per_round)


def test_iterative_initial_return_predicates_match_sklearn_branches() -> None:
    mask = _mask()
    all_missing = np.ones_like(mask, dtype=np.bool_)

    assert iterative_fit_initial_return_required(0, mask) is True
    assert iterative_fit_initial_return_required(4, all_missing) is True
    assert iterative_fit_initial_return_required(4, mask) is False

    assert iterative_transform_initial_return_required(0, mask) is True
    assert iterative_transform_initial_return_required(3, all_missing) is True
    assert iterative_transform_initial_return_required(3, mask) is False


def test_iterative_single_feature_and_missing_feature_count_match_fit_shell() -> None:
    mask = _mask()
    ordered_idx = iterative_ordered_feature_indices(mask, imputation_order="ascending", skip_complete=False, random_state=0)

    assert iterative_single_feature_return_required(1) is True
    assert iterative_single_feature_return_required(2) is False
    assert iterative_missing_feature_count(ordered_idx) == len(ordered_idx)


def test_iterative_require_strict_limits_matches_fit_transform_guard() -> None:
    is_empty_feature = np.array([False, True, False], dtype=np.bool_)
    min_values = iterative_limit_vector(np.array([-1.0, -2.0, 0.0]), is_empty_feature, limit_type="min", keep_empty_features=False)
    max_values = iterative_limit_vector(np.array([2.0, 3.0, 5.0]), is_empty_feature, limit_type="max", keep_empty_features=False)

    validated = iterative_require_strict_limits(min_values, max_values)
    assert np.array_equal(validated, max_values)

    with pytest.raises(ValueError, match="min_value >= max_value"):
        iterative_require_strict_limits(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 2.0], dtype=np.float64),
        )


def test_iterative_normalized_tolerance_matches_private_fit_expression() -> None:
    X = np.array(
        [
            [1.0, np.nan, 3.0],
            [2.0, 4.0, np.nan],
            [3.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    mask = np.isnan(X)
    tol = 1e-3

    observed = iterative_normalized_tolerance(X, mask, tol=tol)
    expected = tol * np.max(np.abs(X[~mask]))
    assert observed == pytest.approx(expected)


def test_iterative_imputations_per_round_matches_transform_expression() -> None:
    imp = IterativeImputer(max_iter=3, random_state=0)
    imp.imputation_sequence_ = [object()] * 12
    imp.n_iter_ = 3

    observed = iterative_imputations_per_round(len(imp.imputation_sequence_), imp.n_iter_)
    expected = len(imp.imputation_sequence_) // imp.n_iter_
    assert observed == expected == 4


def test_iterative_loop_bookkeeping_contracts_reject_invalid_inputs() -> None:
    with pytest.raises(ViolationError):
        iterative_fit_initial_return_required(-1, _mask())

    with pytest.raises(ViolationError):
        iterative_transform_initial_return_required(1, np.array([[1, 0]], dtype=np.int64))

    with pytest.raises(ViolationError):
        iterative_single_feature_return_required(-1)

    with pytest.raises(ViolationError):
        iterative_require_strict_limits(np.array([0.0]), np.array([[1.0]]))

    with pytest.raises(ViolationError):
        iterative_missing_feature_count(np.array([[0, 1]], dtype=np.int64))

    with pytest.raises(ViolationError):
        iterative_normalized_tolerance(np.array([[1.0, 2.0]]), np.array([[True]], dtype=np.bool_), tol=1e-3)

    with pytest.raises(ViolationError):
        iterative_imputations_per_round(5, 2)

