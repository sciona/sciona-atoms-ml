from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model import LarsCV, LassoLarsCV, lars_path as sklearn_lars_path
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import KFold


def _make_regression_fixture() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [1.0, 2.0, -0.5, 1.5],
            [0.0, -1.0, 2.0, 0.25],
            [1.5, 0.5, 0.0, -1.0],
            [2.0, 1.0, 1.0, 0.0],
            [-1.0, 1.5, 0.5, 2.0],
            [0.75, -0.5, 1.25, -1.5],
            [1.25, 0.0, -1.0, 0.5],
            [-0.5, 1.0, 2.0, 1.0],
            [2.5, -1.5, 0.25, 0.0],
        ],
        dtype=np.float64,
    )
    y = np.array([2.5, -0.5, 1.75, 2.0, 0.25, -1.0, 0.5, 1.0, 1.25], dtype=np.float64)
    return X, y


def test_lars_cv_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv import (
        lars_cv_alpha_grid,
        lars_cv_best_alpha,
        lars_cv_finite_row_mask,
        lars_cv_interpolated_fold_mse,
        lars_cv_residual_path,
    )

    assert callable(lars_cv_residual_path)
    assert callable(lars_cv_alpha_grid)
    assert callable(lars_cv_interpolated_fold_mse)
    assert callable(lars_cv_finite_row_mask)
    assert callable(lars_cv_best_alpha)


def test_lars_cv_residual_path_matches_private_helper() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv import lars_cv_residual_path

    X, y = _make_regression_fixture()
    train = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    test = np.array([6, 7, 8], dtype=np.int64)
    X_train = X[train].copy()
    y_train = y[train].copy()
    X_test = X[test].copy()
    y_test = y[test].copy()

    expected_alphas, _, _, expected_residues = _lars_path_residues(
        X_train.copy(),
        y_train.copy(),
        X_test.copy(),
        y_test.copy(),
        method="lar",
        fit_intercept=True,
        max_iter=5,
        copy=True,
    )

    x_mean = X_train.mean(axis=0)
    y_mean = float(y_train.mean())
    centered_train = X_train - x_mean
    centered_target = y_train - y_mean
    result_alphas, _, result_coefs = sklearn_lars_path(
        centered_train,
        centered_target,
        method="lar",
        max_iter=5,
    )
    result_residues = lars_cv_residual_path(X_test, y_test, result_coefs, x_mean, y_mean=y_mean)

    assert np.allclose(result_alphas, expected_alphas)
    assert np.allclose(result_residues, expected_residues)


def test_lars_cv_helpers_reconstruct_larscv_outputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv import (
        lars_cv_alpha_grid,
        lars_cv_best_alpha,
        lars_cv_finite_row_mask,
        lars_cv_interpolated_fold_mse,
    )

    X, y = _make_regression_fixture()
    cv = KFold(n_splits=3)
    cv_paths = [
        _lars_path_residues(
            X[train].copy(),
            y[train].copy(),
            X[test].copy(),
            y[test].copy(),
            method="lar",
            fit_intercept=True,
            max_iter=5,
            copy=True,
        )
        for train, test in cv.split(X, y)
    ]

    shared_alphas = lars_cv_alpha_grid(tuple(path[0] for path in cv_paths), max_n_alphas=1000)
    fold_mse = [
        lars_cv_interpolated_fold_mse(alphas, residues, shared_alphas)
        for alphas, _, _, residues in cv_paths
    ]
    mse_path = np.column_stack(fold_mse)
    finite_mask = lars_cv_finite_row_mask(mse_path)
    filtered_alphas = shared_alphas[finite_mask]
    filtered_mse = mse_path[finite_mask]
    best_alpha = lars_cv_best_alpha(filtered_alphas, filtered_mse)

    expected = LarsCV(cv=cv, max_iter=5, max_n_alphas=1000).fit(X, y)

    assert np.allclose(filtered_alphas, expected.cv_alphas_)
    assert np.allclose(filtered_mse, expected.mse_path_)
    assert best_alpha == pytest.approx(expected.alpha_)


def test_lars_cv_helpers_reconstruct_lasso_lars_cv_outputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv import (
        lars_cv_alpha_grid,
        lars_cv_best_alpha,
        lars_cv_finite_row_mask,
        lars_cv_interpolated_fold_mse,
    )

    X, y = _make_regression_fixture()
    cv = KFold(n_splits=3)
    cv_paths = [
        _lars_path_residues(
            X[train].copy(),
            y[train].copy(),
            X[test].copy(),
            y[test].copy(),
            method="lasso",
            fit_intercept=True,
            max_iter=5,
            copy=True,
        )
        for train, test in cv.split(X, y)
    ]

    shared_alphas = lars_cv_alpha_grid(tuple(path[0] for path in cv_paths), max_n_alphas=1000)
    fold_mse = [
        lars_cv_interpolated_fold_mse(alphas, residues, shared_alphas)
        for alphas, _, _, residues in cv_paths
    ]
    mse_path = np.column_stack(fold_mse)
    finite_mask = lars_cv_finite_row_mask(mse_path)
    filtered_alphas = shared_alphas[finite_mask]
    filtered_mse = mse_path[finite_mask]
    best_alpha = lars_cv_best_alpha(filtered_alphas, filtered_mse)

    expected = LassoLarsCV(cv=cv, max_iter=5, max_n_alphas=1000).fit(X, y)

    assert np.allclose(filtered_alphas, expected.cv_alphas_)
    assert np.allclose(filtered_mse, expected.mse_path_)
    assert best_alpha == pytest.approx(expected.alpha_)


def test_lars_cv_finite_row_mask_marks_only_finite_rows() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv import lars_cv_finite_row_mask

    mse_path = np.array(
        [
            [1.0, 2.0, 3.0],
            [np.nan, 1.0, 2.0],
            [1.0, np.inf, 2.0],
            [0.5, 0.75, 1.0],
        ],
        dtype=np.float64,
    )

    result = lars_cv_finite_row_mask(mse_path)

    assert np.array_equal(result, np.array([True, False, False, True], dtype=np.bool_))


def test_lars_cv_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv import (
        lars_cv_alpha_grid,
        lars_cv_best_alpha,
        lars_cv_interpolated_fold_mse,
        lars_cv_residual_path,
    )

    with pytest.raises(ViolationError):
        lars_cv_alpha_grid((np.array([0.1, -0.2], dtype=np.float64),), max_n_alphas=10)

    with pytest.raises(ViolationError):
        lars_cv_residual_path(
            np.ones((2, 3), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            np.ones((4, 2), dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        lars_cv_interpolated_fold_mse(
            np.array([0.4, 0.2], dtype=np.float64),
            np.ones((3, 2), dtype=np.float64),
            np.array([0.0, 0.2, 0.4], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        lars_cv_best_alpha(
            np.array([0.0, 0.2], dtype=np.float64),
            np.array([[1.0, np.nan], [0.5, 0.4]], dtype=np.float64),
        )
