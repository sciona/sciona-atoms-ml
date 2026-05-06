"""Ghost witnesses for sklearn coordinate-descent CV alpha-bookkeeping atoms."""

from __future__ import annotations


def witness_cd_cv_has_l1_ratio_param(path_params: object) -> object:
    """Describe the `'l1_ratio' in path_params` branch in LinearModelCV.fit."""
    return path_params


def witness_cd_cv_l1_ratios(l1_ratio_value: object) -> object:
    """Describe the `np.atleast_1d(path_params['l1_ratio'])` shell in LinearModelCV.fit."""
    return l1_ratio_value


def witness_cd_cv_first_path_l1_ratio(l1_ratios: object) -> object:
    """Describe the first-path l1_ratio selection in LinearModelCV.fit."""
    return l1_ratios


def witness_cd_cv_default_l1_ratios(has_l1_ratio_param: object) -> object:
    """Describe the default l1_ratio list in LinearModelCV.fit."""
    return has_l1_ratio_param


def witness_cd_cv_alpha_grid_required(alphas_is_none: object) -> object:
    """Describe the `if alphas is None` branch in LinearModelCV.fit."""
    return alphas_is_none


def witness_cd_cv_sorted_alphas(alphas: object, n_l1_ratio: object) -> object:
    """Describe the sorted-and-tiled alpha grid shell in LinearModelCV.fit."""
    return alphas, n_l1_ratio


def witness_cd_cv_n_l1_ratio(l1_ratios: object) -> object:
    """Describe the `len(l1_ratios)` shell in LinearModelCV.fit."""
    return l1_ratios


def witness_cd_cv_n_alphas(alphas: object) -> object:
    """Describe the `len(alphas[0])` shell in LinearModelCV.fit."""
    return alphas
