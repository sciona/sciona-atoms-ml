"""Sklearn coordinate-descent CV MSE-selection atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_alphas_from_auto_grid,
    witness_cd_cv_alphas_from_user_grid,
    witness_cd_cv_best_alpha_index,
    witness_cd_cv_best_alpha_value,
    witness_cd_cv_best_l1_ratio_value,
    witness_cd_cv_best_mse_value,
    witness_cd_cv_mean_mse,
    witness_cd_cv_mse_path_public,
    witness_cd_cv_mse_paths_reshaped,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _nonempty_numeric_1d(value: object) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 1 and value.size >= 1


@register_atom(witness_cd_cv_mse_paths_reshaped)
@icontract.require(lambda mse_paths: isinstance(mse_paths, np.ndarray) and mse_paths.ndim >= 1, "mse_paths must be an ndarray")
@icontract.require(lambda n_l1_ratio: _positive_int(n_l1_ratio), "n_l1_ratio must be positive")
@icontract.require(lambda fold_count: _positive_int(fold_count), "fold_count must be positive")
@icontract.ensure(
    lambda result, n_l1_ratio, fold_count: isinstance(result, np.ndarray)
    and result.shape[0] == int(n_l1_ratio)
    and result.shape[1] == int(fold_count),
    "reshaped mse_paths must use (n_l1_ratio, fold_count, -1)",
)
def cd_cv_mse_paths_reshaped(
    mse_paths: np.ndarray, n_l1_ratio: int, fold_count: int
) -> np.ndarray:
    """Return mse_paths reshaped to (n_l1_ratio, fold_count, -1)."""
    return np.reshape(mse_paths, (int(n_l1_ratio), int(fold_count), -1))


@register_atom(witness_cd_cv_mean_mse)
@icontract.require(
    lambda mse_paths_reshaped: isinstance(mse_paths_reshaped, np.ndarray) and mse_paths_reshaped.ndim == 3,
    "mse_paths_reshaped must be a 3D ndarray",
)
@icontract.ensure(
    lambda result, mse_paths_reshaped: isinstance(result, np.ndarray)
    and result.shape == (mse_paths_reshaped.shape[0], mse_paths_reshaped.shape[2]),
    "mean_mse must average over folds and preserve l1_ratio x alpha layout",
)
def cd_cv_mean_mse(mse_paths_reshaped: np.ndarray) -> np.ndarray:
    """Return the mean MSE averaged across folds."""
    return np.mean(mse_paths_reshaped, axis=1)


@register_atom(witness_cd_cv_mse_path_public)
@icontract.require(
    lambda mse_paths_reshaped: isinstance(mse_paths_reshaped, np.ndarray) and mse_paths_reshaped.ndim == 3,
    "mse_paths_reshaped must be a 3D ndarray",
)
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "public mse_path_ must be an ndarray")
def cd_cv_mse_path_public(mse_paths_reshaped: np.ndarray) -> np.ndarray:
    """Return the public mse_path_ packaging used by LinearModelCV.fit."""
    return np.squeeze(np.moveaxis(mse_paths_reshaped, 2, 1))


@register_atom(witness_cd_cv_best_alpha_index)
@icontract.require(
    lambda mse_alphas: _nonempty_numeric_1d(mse_alphas),
    "mse_alphas must be a nonempty 1D ndarray",
)
@icontract.ensure(
    lambda result, mse_alphas: _positive_int(result) and 0 <= int(result) < len(mse_alphas),
    "best alpha index must be a valid argmin position",
)
def cd_cv_best_alpha_index(mse_alphas: np.ndarray) -> int:
    """Return the best alpha index selected by argmin."""
    return int(np.argmin(mse_alphas))


@register_atom(witness_cd_cv_best_mse_value)
@icontract.require(
    lambda mse_alphas: _nonempty_numeric_1d(mse_alphas),
    "mse_alphas must be a nonempty 1D ndarray",
)
@icontract.require(
    lambda best_alpha_index, mse_alphas: _positive_int(best_alpha_index)
    and 0 <= int(best_alpha_index) < len(mse_alphas),
    "best_alpha_index must be a valid position in mse_alphas",
)
@icontract.ensure(
    lambda result, mse_alphas, best_alpha_index: np.isclose(float(result), float(mse_alphas[int(best_alpha_index)])),
    "best MSE must equal mse_alphas[best_alpha_index]",
)
def cd_cv_best_mse_value(mse_alphas: np.ndarray, best_alpha_index: int) -> float:
    """Return the best MSE value selected from one l1_ratio path."""
    return float(mse_alphas[int(best_alpha_index)])


@register_atom(witness_cd_cv_best_alpha_value)
@icontract.require(
    lambda l1_alphas: _nonempty_numeric_1d(l1_alphas),
    "l1_alphas must be a nonempty 1D ndarray",
)
@icontract.require(
    lambda best_alpha_index, l1_alphas: _positive_int(best_alpha_index)
    and 0 <= int(best_alpha_index) < len(l1_alphas),
    "best_alpha_index must be a valid position in l1_alphas",
)
@icontract.ensure(
    lambda result, l1_alphas, best_alpha_index: np.isclose(float(result), float(l1_alphas[int(best_alpha_index)])),
    "best alpha must equal l1_alphas[best_alpha_index]",
)
def cd_cv_best_alpha_value(l1_alphas: np.ndarray, best_alpha_index: int) -> float:
    """Return the best alpha selected from one l1_ratio path."""
    return float(l1_alphas[int(best_alpha_index)])


@register_atom(witness_cd_cv_best_l1_ratio_value)
@icontract.ensure(lambda result, l1_ratio: result == l1_ratio, "best l1_ratio value must pass through unchanged")
def cd_cv_best_l1_ratio_value(l1_ratio: object) -> object:
    """Return the best l1_ratio value selected for refit."""
    return l1_ratio


@register_atom(witness_cd_cv_alphas_from_auto_grid)
@icontract.require(
    lambda alphas: hasattr(alphas, "__len__") and len(alphas) >= 1,
    "alphas must contain at least one grid",
)
@icontract.require(lambda n_l1_ratio: _positive_int(n_l1_ratio), "n_l1_ratio must be positive")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "auto-grid alphas_ must be an ndarray")
def cd_cv_alphas_from_auto_grid(alphas: object, n_l1_ratio: int) -> np.ndarray:
    """Return alphas_ packaging when self.alphas is None."""
    result = np.asarray(alphas)
    if int(n_l1_ratio) == 1:
        return result[0]
    return result


@register_atom(witness_cd_cv_alphas_from_user_grid)
@icontract.require(
    lambda alphas: hasattr(alphas, "__getitem__") and len(alphas) >= 1,
    "alphas must contain at least one grid",
)
@icontract.ensure(
    lambda result, alphas: isinstance(result, np.ndarray)
    and np.array_equal(result, np.asarray(alphas[0])),
    "user-grid alphas_ must equal np.asarray(alphas[0])",
)
def cd_cv_alphas_from_user_grid(alphas: object) -> np.ndarray:
    """Return alphas_ packaging when user alphas are provided."""
    return np.asarray(alphas[0])
