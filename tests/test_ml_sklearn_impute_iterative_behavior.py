from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def test_iterative_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative import (
        iterative_convergence_reached,
        iterative_limit_vector,
        iterative_neighbor_feature_indices,
        iterative_normalized_abs_corr_matrix,
        iterative_ordered_feature_indices,
    )

    assert callable(iterative_ordered_feature_indices)
    assert callable(iterative_normalized_abs_corr_matrix)
    assert callable(iterative_neighbor_feature_indices)
    assert callable(iterative_limit_vector)
    assert callable(iterative_convergence_reached)


def test_iterative_ordered_feature_indices_matches_sklearn_orders() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative import iterative_ordered_feature_indices

    mask = np.array(
        [
            [True, False, True, False],
            [False, False, True, True],
            [False, False, False, True],
        ],
        dtype=np.bool_,
    )
    for order in ["roman", "arabic", "ascending", "descending"]:
        imp = IterativeImputer(imputation_order=order, skip_complete=True)
        expected = imp._get_ordered_idx(mask)
        result = iterative_ordered_feature_indices(mask, imputation_order=order, skip_complete=True)
        assert np.array_equal(result, expected)


def test_iterative_ordered_feature_indices_random_matches_seeded_sklearn() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative import iterative_ordered_feature_indices

    mask = np.array([[True, False, True, True], [False, True, False, True]], dtype=np.bool_)
    imp = IterativeImputer(imputation_order="random", skip_complete=False, random_state=42)
    imp.random_state_ = np.random.RandomState(42)

    assert np.array_equal(
        iterative_ordered_feature_indices(mask, imputation_order="random", skip_complete=False, random_state=42),
        imp._get_ordered_idx(mask),
    )


def test_iterative_normalized_abs_corr_matrix_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative import iterative_normalized_abs_corr_matrix

    X_filled = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 1.0],
            [3.0, 6.0, 1.0],
            [4.0, 8.0, 1.0],
        ],
        dtype=np.float64,
    )
    imp = IterativeImputer(n_nearest_features=2)
    expected = imp._get_abs_corr_mat(X_filled)
    result = iterative_normalized_abs_corr_matrix(X_filled)

    assert np.allclose(result, expected)
    assert np.allclose(result.sum(axis=0), 1.0)
    assert np.all(result.diagonal() == 0.0)


def test_iterative_neighbor_feature_indices_matches_full_and_sampled_sklearn() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative import (
        iterative_neighbor_feature_indices,
        iterative_normalized_abs_corr_matrix,
    )

    X_filled = np.array(
        [[1.0, 2.0, 1.0, 4.0], [2.0, 3.0, 1.0, 5.0], [3.0, 5.0, 1.0, 7.0], [4.0, 8.0, 1.0, 11.0]],
        dtype=np.float64,
    )
    abs_corr = iterative_normalized_abs_corr_matrix(X_filled)

    assert np.array_equal(iterative_neighbor_feature_indices(4, 2, None), np.array([0, 1, 3], dtype=np.int64))

    imp = IterativeImputer(n_nearest_features=2, random_state=7)
    imp.random_state_ = np.random.RandomState(7)
    expected = imp._get_neighbor_feat_idx(4, 2, abs_corr)
    result = iterative_neighbor_feature_indices(4, 2, 2, abs_corr_mat=abs_corr, random_state=7)
    assert np.array_equal(result, expected)


def test_iterative_limit_vector_matches_sklearn_validation() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative import iterative_limit_vector

    is_empty = np.array([False, True, False], dtype=np.bool_)

    expected_scalar = IterativeImputer._validate_limit(0.5, "min", 3, is_empty, False)
    assert np.allclose(iterative_limit_vector(0.5, is_empty, limit_type="min", keep_empty_features=False), expected_scalar)

    vector_limit = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    expected_vector = IterativeImputer._validate_limit(vector_limit, "max", 3, is_empty, False)
    assert np.allclose(iterative_limit_vector(vector_limit, is_empty, limit_type="max", keep_empty_features=False), expected_vector)

    expected_none = IterativeImputer._validate_limit(None, "max", 3, is_empty, True)
    assert np.allclose(iterative_limit_vector(None, is_empty, limit_type="max", keep_empty_features=True), expected_none)


def test_iterative_convergence_reached_matches_source_formula() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative import iterative_convergence_reached

    X_previous = np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float64)
    X_current = np.array([[1.0, 2.01], [4.0, 5.0]], dtype=np.float64)
    X_original = np.array([[1.0, np.nan], [4.0, 5.0]], dtype=np.float64)
    mask = np.isnan(X_original)

    observed_scale = np.max(np.abs(X_original[~mask]))
    change = np.linalg.norm(X_current - X_previous, ord=np.inf, axis=None)
    assert iterative_convergence_reached(X_current, X_previous, X_original, mask, tol=0.01) is bool(change < 0.01 * observed_scale)
    assert iterative_convergence_reached(X_current, X_previous, X_original, mask, tol=0.0001) is False


def test_contracts_reject_invalid_iterative_inputs() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative import (
        iterative_limit_vector,
        iterative_neighbor_feature_indices,
        iterative_normalized_abs_corr_matrix,
        iterative_ordered_feature_indices,
    )

    with pytest.raises(ViolationError):
        iterative_ordered_feature_indices(np.array([[1, 0]], dtype=np.int64))

    with pytest.raises(ViolationError):
        iterative_normalized_abs_corr_matrix(np.ones((3, 1), dtype=np.float64))

    with pytest.raises(ViolationError):
        iterative_neighbor_feature_indices(3, 1, 2, abs_corr_mat=None)

    with pytest.raises(ViolationError):
        iterative_limit_vector(np.array([1.0, 2.0], dtype=np.float64), np.array([False, True, False]), limit_type="max", keep_empty_features=True)
