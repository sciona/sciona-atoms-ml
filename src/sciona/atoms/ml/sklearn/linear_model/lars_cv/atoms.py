"""LARS cross-validation helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import interpolate

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_lars_cv_alpha_grid,
    witness_lars_cv_best_alpha,
    witness_lars_cv_finite_row_mask,
    witness_lars_cv_interpolated_fold_mse,
    witness_lars_cv_residual_path,
)


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_vector(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _nonnegative_vector(values: NDArray[np.float64]) -> bool:
    return bool(_finite_vector(values) and np.all(np.asarray(values, dtype=np.float64) >= 0.0))


def _finite_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _test_inputs_valid(
    X_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    coefs: NDArray[np.float64],
    x_mean: NDArray[np.float64],
) -> bool:
    x_values = np.asarray(X_test, dtype=np.float64)
    y_values = np.asarray(y_test, dtype=np.float64)
    coef_values = np.asarray(coefs, dtype=np.float64)
    mean_values = np.asarray(x_mean, dtype=np.float64)
    return bool(
        _finite_matrix(X_test)
        and _finite_vector(y_test)
        and _finite_matrix(coefs)
        and _finite_vector(x_mean)
        and x_values.shape[0] == y_values.shape[0]
        and x_values.shape[1] == coef_values.shape[0]
        and x_values.shape[1] == mean_values.shape[0]
    )


def _fold_alphas_valid(fold_alphas: NDArray[np.float64]) -> bool:
    return _nonnegative_vector(fold_alphas)


def _fold_alpha_collection_valid(fold_alphas: tuple[NDArray[np.float64], ...]) -> bool:
    return bool(
        isinstance(fold_alphas, tuple)
        and len(fold_alphas) >= 1
        and all(_fold_alphas_valid(values) for values in fold_alphas)
    )


def _target_alphas_valid(target_alphas: NDArray[np.float64]) -> bool:
    values = np.asarray(target_alphas, dtype=np.float64)
    return bool(
        _nonnegative_vector(target_alphas)
        and len(np.unique(values)) == values.shape[0]
        and (values.shape[0] == 1 or np.all(np.diff(values) > 0.0))
    )


def _fold_residue_inputs_valid(
    fold_alphas: NDArray[np.float64],
    fold_residues: NDArray[np.float64],
    target_alphas: NDArray[np.float64],
) -> bool:
    alphas = np.asarray(fold_alphas, dtype=np.float64)
    residues = np.asarray(fold_residues, dtype=np.float64)
    return bool(
        _fold_alphas_valid(fold_alphas)
        and _finite_matrix(fold_residues)
        and _target_alphas_valid(target_alphas)
        and residues.shape[0] == alphas.shape[0]
    )


def _mse_path_valid(cv_alphas: NDArray[np.float64], mse_path: NDArray[np.float64]) -> bool:
    alpha_values = np.asarray(cv_alphas, dtype=np.float64)
    mse_values = np.asarray(mse_path, dtype=np.float64)
    return bool(
        _target_alphas_valid(cv_alphas)
        and mse_values.ndim == 2
        and mse_values.shape[0] == alpha_values.shape[0]
        and mse_values.shape[1] >= 1
    )


def _finite_mse_path_valid(cv_alphas: NDArray[np.float64], mse_path: NDArray[np.float64]) -> bool:
    return bool(_mse_path_valid(cv_alphas, mse_path) and np.all(np.isfinite(np.asarray(mse_path, dtype=np.float64))))


def _residual_path_result_valid(result: NDArray[np.float64], X_test: NDArray[np.float64], coefs: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    x_values = np.asarray(X_test, dtype=np.float64)
    coef_values = np.asarray(coefs, dtype=np.float64)
    return bool(values.shape == (coef_values.shape[1], x_values.shape[0]) and np.all(np.isfinite(values)))


def _alpha_grid_result_valid(result: NDArray[np.float64], fold_alphas: tuple[NDArray[np.float64], ...]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    concatenated = np.concatenate(fold_alphas)
    return bool(
        values.ndim == 1
        and values.shape[0] >= 1
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and len(np.unique(values)) == values.shape[0]
        and (values.shape[0] == 1 or np.all(np.diff(values) > 0.0))
        and set(values.tolist()).issubset(set(np.unique(concatenated).tolist()))
    )


def _fold_mse_result_valid(result: NDArray[np.float64], target_alphas: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    targets = np.asarray(target_alphas, dtype=np.float64)
    return bool(values.shape == targets.shape and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _mask_result_valid(result: NDArray[np.bool_], cv_alphas: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    alpha_values = np.asarray(cv_alphas, dtype=np.float64)
    return bool(values.dtype == np.bool_ and values.shape == alpha_values.shape)


def _best_alpha_result_valid(result: float, cv_alphas: NDArray[np.float64]) -> bool:
    values = np.asarray(cv_alphas, dtype=np.float64)
    return bool(np.isfinite(float(result)) and float(result) in set(values.tolist()))


@register_atom(witness_lars_cv_residual_path)
@icontract.require(lambda X_test, y_test, coefs, x_mean: _test_inputs_valid(X_test, y_test, coefs, x_mean), "test design, targets, coefficient path, and training means must align")
@icontract.require(lambda y_mean: _finite_scalar(y_mean), "y_mean must be finite")
@icontract.ensure(lambda result, X_test, coefs: _residual_path_result_valid(result, X_test, coefs), "residual path must provide one finite test-residual vector per alpha")
def lars_cv_residual_path(
    X_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    coefs: NDArray[np.float64],
    x_mean: NDArray[np.float64],
    *,
    y_mean: float = 0.0,
) -> NDArray[np.float64]:
    """Compute per-alpha left-out residuals from supplied LARS path coefficients and training means."""
    x_values = np.asarray(X_test, dtype=np.float64)
    y_values = np.asarray(y_test, dtype=np.float64)
    coef_values = np.asarray(coefs, dtype=np.float64)
    centered_x = x_values - np.asarray(x_mean, dtype=np.float64)
    centered_y = y_values - float(y_mean)
    residues = centered_x.dot(coef_values) - centered_y[:, np.newaxis]
    return np.asarray(residues.T, dtype=np.float64)


@register_atom(witness_lars_cv_alpha_grid)
@icontract.require(lambda fold_alphas: _fold_alpha_collection_valid(fold_alphas), "fold_alphas must be a nonempty tuple of finite nonnegative alpha paths")
@icontract.require(lambda max_n_alphas: _positive_int(max_n_alphas), "max_n_alphas must be positive")
@icontract.ensure(lambda result, fold_alphas: _alpha_grid_result_valid(result, fold_alphas), "alpha grid must be a finite sorted unique subset of fold alphas")
def lars_cv_alpha_grid(
    fold_alphas: tuple[NDArray[np.float64], ...],
    *,
    max_n_alphas: int = 1000,
) -> NDArray[np.float64]:
    """Build sklearn's shared sorted cross-validation alpha grid from fold alpha paths."""
    all_alphas = np.unique(np.concatenate(tuple(np.asarray(values, dtype=np.float64) for values in fold_alphas)))
    stride = int(max(1, int(len(all_alphas) / float(max_n_alphas))))
    return np.asarray(all_alphas[::stride], dtype=np.float64)


@register_atom(witness_lars_cv_interpolated_fold_mse)
@icontract.require(lambda fold_alphas, fold_residues, target_alphas: _fold_residue_inputs_valid(fold_alphas, fold_residues, target_alphas), "fold alphas, residues, and target grid must align and follow sklearn path ordering")
@icontract.ensure(lambda result, target_alphas: _fold_mse_result_valid(result, target_alphas), "fold MSE must contain one finite nonnegative value per target alpha")
def lars_cv_interpolated_fold_mse(
    fold_alphas: NDArray[np.float64],
    fold_residues: NDArray[np.float64],
    target_alphas: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project one fold's residual path onto a shared alpha grid and average squared residues over test samples."""
    alphas = np.asarray(fold_alphas, dtype=np.float64)[::-1]
    residues = np.asarray(fold_residues, dtype=np.float64)[::-1]
    targets = np.asarray(target_alphas, dtype=np.float64)

    if alphas[0] != 0.0:
        alphas = np.r_[0.0, alphas]
        residues = np.r_[residues[0, np.newaxis], residues]
    if alphas[-1] != targets[-1]:
        alphas = np.r_[alphas, targets[-1]]
        residues = np.r_[residues, residues[-1, np.newaxis]]

    projected = interpolate.interp1d(alphas, residues, axis=0)(targets)
    projected = np.asarray(projected, dtype=np.float64)
    projected **= 2
    return np.asarray(np.mean(projected, axis=-1), dtype=np.float64)


@register_atom(witness_lars_cv_finite_row_mask)
@icontract.require(lambda mse_path: _mse_path_valid(np.arange(np.asarray(mse_path).shape[0], dtype=np.float64), mse_path), "mse_path must be a nonempty alpha-by-fold matrix")
@icontract.ensure(lambda result, mse_path: _mask_result_valid(result, np.arange(np.asarray(mse_path).shape[0], dtype=np.float64)), "mask must mark one row per alpha")
def lars_cv_finite_row_mask(mse_path: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Mark alpha rows whose fold MSE values are finite across every fold."""
    return np.asarray(np.all(np.isfinite(np.asarray(mse_path, dtype=np.float64)), axis=-1), dtype=np.bool_)


@register_atom(witness_lars_cv_best_alpha)
@icontract.require(lambda cv_alphas, mse_path: _finite_mse_path_valid(cv_alphas, mse_path), "cv_alphas and mse_path must align and be finite after filtering")
@icontract.ensure(lambda result, cv_alphas: _best_alpha_result_valid(result, cv_alphas), "best alpha must be selected from cv_alphas")
def lars_cv_best_alpha(
    cv_alphas: NDArray[np.float64],
    mse_path: NDArray[np.float64],
) -> float:
    """Select the shared alpha with the smallest mean left-out MSE across folds."""
    alpha_values = np.asarray(cv_alphas, dtype=np.float64)
    mse_values = np.asarray(mse_path, dtype=np.float64)
    return float(alpha_values[int(np.argmin(np.mean(mse_values, axis=-1)))])
