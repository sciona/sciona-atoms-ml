"""Ghost witnesses for sklearn coordinate-descent CV path-parameter atoms."""

from __future__ import annotations


def witness_cd_cv_path_params_fit_intercept_removed(path_params: object) -> object:
    """Describe the fit_intercept pop shell in LinearModelCV.fit."""
    return path_params


def witness_cd_cv_path_params_cv_removed(path_params: object) -> object:
    """Describe the cv pop shell in LinearModelCV.fit."""
    return path_params


def witness_cd_cv_path_params_n_jobs_removed(path_params: object) -> object:
    """Describe the n_jobs pop shell in LinearModelCV.fit."""
    return path_params


def witness_cd_cv_path_params_n_alphas(n_alphas: object) -> object:
    """Describe the n_alphas path-parameter update in LinearModelCV.fit."""
    return n_alphas


def witness_cd_cv_path_params_copy_x(copy_x: object) -> object:
    """Describe the initial copy_X path-parameter assignment in LinearModelCV.fit."""
    return copy_x


def witness_cd_cv_parallel_copy_x_override_required(effective_n_jobs_gt_one: object) -> object:
    """Describe the parallel copy_X override guard in LinearModelCV.fit."""
    return effective_n_jobs_gt_one


def witness_cd_cv_resolved_path_copy_x(
    initial_copy_x: object, parallel_override_required: object
) -> object:
    """Describe the resolved copy_X path-parameter value in LinearModelCV.fit."""
    return initial_copy_x, parallel_override_required
