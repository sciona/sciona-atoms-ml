"""Ghost witnesses for sklearn coordinate-descent path-residual path-parameter atoms."""

from __future__ import annotations


def witness_cd_path_residuals_prefit_copy_flag(fit_intercept: object) -> object:
    """Describe the fixed copy=False kwarg passed into _pre_fit."""
    return fit_intercept


def witness_cd_path_residuals_path_params_Xy(Xy: object) -> object:
    """Describe the Xy passthrough inserted into path_params."""
    return Xy


def witness_cd_path_residuals_path_params_X_offset(X_offset: object) -> object:
    """Describe the X_offset passthrough inserted into path_params."""
    return X_offset


def witness_cd_path_residuals_path_params_X_scale(X_scale: object) -> object:
    """Describe the X_scale passthrough inserted into path_params."""
    return X_scale


def witness_cd_path_residuals_path_params_precompute(precompute: object) -> object:
    """Describe the resolved precompute value inserted into path_params."""
    return precompute


def witness_cd_path_residuals_path_params_copy_x(path_params: object) -> object:
    """Describe the fixed copy_X=False assignment in path_params."""
    return path_params


def witness_cd_path_residuals_path_params_alphas(alphas: object) -> object:
    """Describe the alphas passthrough inserted into path_params."""
    return alphas


def witness_cd_path_residuals_path_params_sample_weight(train_sample_weight: object) -> object:
    """Describe the sample_weight passthrough inserted into path_params."""
    return train_sample_weight


def witness_cd_path_residuals_l1_ratio_update_required(path_params: object) -> object:
    """Describe the conditional l1_ratio update gate in _path_residuals."""
    return path_params


def witness_cd_path_residuals_path_params_l1_ratio(l1_ratio: object) -> object:
    """Describe the l1_ratio passthrough inserted into path_params."""
    return l1_ratio
