"""Sklearn coordinate-descent CV path-parameter atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_parallel_copy_x_override_required,
    witness_cd_cv_path_params_copy_x,
    witness_cd_cv_path_params_cv_removed,
    witness_cd_cv_path_params_fit_intercept_removed,
    witness_cd_cv_path_params_n_alphas,
    witness_cd_cv_path_params_n_jobs_removed,
    witness_cd_cv_resolved_path_copy_x,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and value >= 1


@register_atom(witness_cd_cv_path_params_fit_intercept_removed)
@icontract.require(lambda path_params: isinstance(path_params, dict), "path_params must be a dict")
@icontract.ensure(
    lambda result, path_params: _bool(result) and result == ("fit_intercept" not in path_params),
    "fit_intercept removal flag must reflect its absence from path_params",
)
def cd_cv_path_params_fit_intercept_removed(path_params: dict[object, object]) -> bool:
    """Return whether fit_intercept is absent from the path-parameter mapping."""
    return "fit_intercept" not in path_params


@register_atom(witness_cd_cv_path_params_cv_removed)
@icontract.require(lambda path_params: isinstance(path_params, dict), "path_params must be a dict")
@icontract.ensure(
    lambda result, path_params: _bool(result) and result == ("cv" not in path_params),
    "cv removal flag must reflect its absence from path_params",
)
def cd_cv_path_params_cv_removed(path_params: dict[object, object]) -> bool:
    """Return whether cv is absent from the path-parameter mapping."""
    return "cv" not in path_params


@register_atom(witness_cd_cv_path_params_n_jobs_removed)
@icontract.require(lambda path_params: isinstance(path_params, dict), "path_params must be a dict")
@icontract.ensure(
    lambda result, path_params: _bool(result) and result == ("n_jobs" not in path_params),
    "n_jobs removal flag must reflect its absence from path_params",
)
def cd_cv_path_params_n_jobs_removed(path_params: dict[object, object]) -> bool:
    """Return whether n_jobs is absent from the path-parameter mapping."""
    return "n_jobs" not in path_params


@register_atom(witness_cd_cv_path_params_n_alphas)
@icontract.require(lambda n_alphas: _positive_int(n_alphas), "n_alphas must be positive")
@icontract.ensure(
    lambda result, n_alphas: isinstance(result, dict) and result == {"n_alphas": n_alphas},
    "n_alphas update payload must match path_params.update({'n_alphas': n_alphas})",
)
def cd_cv_path_params_n_alphas(n_alphas: int) -> dict[str, int]:
    """Return the n_alphas update payload applied to path_params."""
    return {"n_alphas": n_alphas}


@register_atom(witness_cd_cv_path_params_copy_x)
@icontract.require(lambda copy_x: _bool(copy_x), "copy_x must be boolean")
@icontract.ensure(
    lambda result, copy_x: _bool(result) and result == copy_x,
    "initial path copy_X must match the estimator-side copy_X value",
)
def cd_cv_path_params_copy_x(copy_x: bool) -> bool:
    """Return the initial copy_X value stored in path_params."""
    return copy_x


@register_atom(witness_cd_cv_parallel_copy_x_override_required)
@icontract.require(
    lambda effective_n_jobs_gt_one: _bool(effective_n_jobs_gt_one),
    "effective_n_jobs_gt_one must be boolean",
)
@icontract.ensure(
    lambda result, effective_n_jobs_gt_one: _bool(result)
    and result == effective_n_jobs_gt_one,
    "parallel copy_X override guard must match effective_n_jobs(self.n_jobs) > 1",
)
def cd_cv_parallel_copy_x_override_required(effective_n_jobs_gt_one: bool) -> bool:
    """Return whether LinearModelCV.fit overrides path_params['copy_X'] to False."""
    return effective_n_jobs_gt_one


@register_atom(witness_cd_cv_resolved_path_copy_x)
@icontract.require(lambda initial_copy_x: _bool(initial_copy_x), "initial_copy_x must be boolean")
@icontract.require(
    lambda parallel_override_required: _bool(parallel_override_required),
    "parallel_override_required must be boolean",
)
@icontract.ensure(
    lambda result, initial_copy_x, parallel_override_required: _bool(result)
    and result == (False if parallel_override_required else initial_copy_x),
    "resolved path copy_X must match sklearn's override logic",
)
def cd_cv_resolved_path_copy_x(
    initial_copy_x: bool, parallel_override_required: bool
) -> bool:
    """Return the final copy_X value stored in path_params after the parallel override."""
    return False if parallel_override_required else initial_copy_x
