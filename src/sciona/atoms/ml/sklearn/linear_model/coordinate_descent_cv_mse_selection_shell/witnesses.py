"""Ghost witnesses for sklearn coordinate-descent CV MSE-selection atoms."""

from __future__ import annotations


def witness_cd_cv_mse_paths_reshaped(mse_paths: object, n_l1_ratio: object, fold_count: object) -> object:
    """Describe the mse_paths reshape shell in LinearModelCV.fit."""
    return mse_paths, n_l1_ratio, fold_count


def witness_cd_cv_mean_mse(mse_paths_reshaped: object) -> object:
    """Describe the mean-over-folds shell in LinearModelCV.fit."""
    return mse_paths_reshaped


def witness_cd_cv_mse_path_public(mse_paths_reshaped: object) -> object:
    """Describe the public mse_path_ packaging shell in LinearModelCV.fit."""
    return mse_paths_reshaped


def witness_cd_cv_best_alpha_index(mse_alphas: object) -> object:
    """Describe the argmin shell for best alpha selection in LinearModelCV.fit."""
    return mse_alphas


def witness_cd_cv_best_mse_value(mse_alphas: object, best_alpha_index: object) -> object:
    """Describe the best-MSE extraction shell in LinearModelCV.fit."""
    return mse_alphas, best_alpha_index


def witness_cd_cv_best_alpha_value(l1_alphas: object, best_alpha_index: object) -> object:
    """Describe the best-alpha extraction shell in LinearModelCV.fit."""
    return l1_alphas, best_alpha_index


def witness_cd_cv_best_l1_ratio_value(l1_ratio: object) -> object:
    """Describe the best-l1-ratio assignment shell in LinearModelCV.fit."""
    return l1_ratio


def witness_cd_cv_alphas_from_auto_grid(alphas: object, n_l1_ratio: object) -> object:
    """Describe the alphas_ packaging shell when self.alphas is None."""
    return alphas, n_l1_ratio


def witness_cd_cv_alphas_from_user_grid(alphas: object) -> object:
    """Describe the alphas_ packaging shell when user alphas are provided."""
    return alphas
