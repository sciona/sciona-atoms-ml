"""Sklearn coordinate-descent path-residual path-parameter atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_path_residuals_l1_ratio_update_required,
    witness_cd_path_residuals_path_params_alphas,
    witness_cd_path_residuals_path_params_copy_x,
    witness_cd_path_residuals_path_params_l1_ratio,
    witness_cd_path_residuals_path_params_precompute,
    witness_cd_path_residuals_path_params_sample_weight,
    witness_cd_path_residuals_path_params_X_offset,
    witness_cd_path_residuals_path_params_X_scale,
    witness_cd_path_residuals_path_params_Xy,
    witness_cd_path_residuals_prefit_copy_flag,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_cd_path_residuals_prefit_copy_flag)
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.ensure(lambda result: _bool(result) and result is False, "_pre_fit copy flag must be fixed to False")
def cd_path_residuals_prefit_copy_flag(fit_intercept: bool) -> bool:
    """Return the fixed copy flag passed to _pre_fit by _path_residuals."""
    del fit_intercept
    return False


@register_atom(witness_cd_path_residuals_path_params_Xy)
@icontract.ensure(lambda result, Xy: result is Xy, "path_params['Xy'] must preserve the _pre_fit output object")
def cd_path_residuals_path_params_Xy(Xy: object) -> object:
    """Return the Xy object inserted into path_params before path execution."""
    return Xy


@register_atom(witness_cd_path_residuals_path_params_X_offset)
@icontract.ensure(
    lambda result, X_offset: result is X_offset,
    "path_params['X_offset'] must preserve the _pre_fit output object",
)
def cd_path_residuals_path_params_X_offset(X_offset: object) -> object:
    """Return the X_offset object inserted into path_params before path execution."""
    return X_offset


@register_atom(witness_cd_path_residuals_path_params_X_scale)
@icontract.ensure(
    lambda result, X_scale: result is X_scale,
    "path_params['X_scale'] must preserve the _pre_fit output object",
)
def cd_path_residuals_path_params_X_scale(X_scale: object) -> object:
    """Return the X_scale object inserted into path_params before path execution."""
    return X_scale


@register_atom(witness_cd_path_residuals_path_params_precompute)
@icontract.ensure(
    lambda result, precompute: result is precompute,
    "path_params['precompute'] must preserve the resolved precompute object",
)
def cd_path_residuals_path_params_precompute(precompute: object) -> object:
    """Return the resolved precompute value inserted into path_params."""
    return precompute


@register_atom(witness_cd_path_residuals_path_params_copy_x)
@icontract.require(lambda path_params: isinstance(path_params, Mapping), "path_params must be a mapping")
@icontract.ensure(
    lambda result: _bool(result) and result is False,
    "path_params['copy_X'] must be fixed to False in _path_residuals",
)
def cd_path_residuals_path_params_copy_x(path_params: Mapping[object, object]) -> bool:
    """Return the fixed copy_X value inserted into path_params."""
    del path_params
    return False


@register_atom(witness_cd_path_residuals_path_params_alphas)
@icontract.ensure(
    lambda result, alphas: result is alphas,
    "path_params['alphas'] must preserve the supplied alphas object",
)
def cd_path_residuals_path_params_alphas(alphas: object) -> object:
    """Return the alphas object inserted into path_params."""
    return alphas


@register_atom(witness_cd_path_residuals_path_params_sample_weight)
@icontract.ensure(
    lambda result, train_sample_weight: result is train_sample_weight,
    "path_params['sample_weight'] must preserve the training sample-weight object",
)
def cd_path_residuals_path_params_sample_weight(train_sample_weight: object) -> object:
    """Return the training sample-weight object inserted into path_params."""
    return train_sample_weight


@register_atom(witness_cd_path_residuals_l1_ratio_update_required)
@icontract.require(lambda path_params: isinstance(path_params, Mapping), "path_params must be a mapping")
@icontract.ensure(
    lambda result, path_params: _bool(result) and result == ("l1_ratio" in path_params),
    "l1_ratio update gate must match membership in path_params",
)
def cd_path_residuals_l1_ratio_update_required(path_params: Mapping[object, object]) -> bool:
    """Return whether _path_residuals updates path_params['l1_ratio']."""
    return "l1_ratio" in path_params


@register_atom(witness_cd_path_residuals_path_params_l1_ratio)
@icontract.ensure(
    lambda result, l1_ratio: result is l1_ratio,
    "path_params['l1_ratio'] must preserve the supplied l1_ratio object",
)
def cd_path_residuals_path_params_l1_ratio(l1_ratio: object) -> object:
    """Return the l1_ratio object inserted into path_params when requested."""
    return l1_ratio
