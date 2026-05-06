"""Sklearn coordinate-descent CV refit-setup atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_refit_common_params,
    witness_cd_cv_refit_copy_x,
    witness_cd_cv_refit_fit_call_uses_sample_weight,
    witness_cd_cv_refit_model_alpha,
    witness_cd_cv_refit_model_l1_ratio,
    witness_cd_cv_refit_precompute_auto_guard_required,
    witness_cd_cv_refit_precompute_value,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _finite_scalar(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


@register_atom(witness_cd_cv_refit_common_params)
@icontract.require(lambda self_params: isinstance(self_params, dict), "self_params must be a dict")
@icontract.require(
    lambda model_param_names: isinstance(model_param_names, (set, frozenset)),
    "model_param_names must be a set of parameter names",
)
@icontract.ensure(
    lambda result, self_params, model_param_names: isinstance(result, dict)
    and set(result.keys()).issubset(model_param_names)
    and all(result[name] == self_params[name] for name in result),
    "common_params must preserve only shared parameter entries",
)
def cd_cv_refit_common_params(
    self_params: dict[object, object], model_param_names: set[str] | frozenset[str]
) -> dict[object, object]:
    """Return the common parameter mapping transferred to the refit model."""
    return {
        name: value
        for name, value in self_params.items()
        if name in model_param_names
    }


@register_atom(witness_cd_cv_refit_model_alpha)
@icontract.require(lambda best_alpha: _finite_scalar(best_alpha), "best_alpha must be finite")
@icontract.ensure(
    lambda result, best_alpha: np.isclose(float(result), float(best_alpha)),
    "model alpha assignment must preserve the selected alpha",
)
def cd_cv_refit_model_alpha(best_alpha: float) -> float:
    """Return the selected alpha assigned to the refit model."""
    return float(best_alpha)


@register_atom(witness_cd_cv_refit_model_l1_ratio)
@icontract.ensure(
    lambda result, best_l1_ratio: result == best_l1_ratio,
    "model l1_ratio assignment must preserve the selected l1_ratio",
)
def cd_cv_refit_model_l1_ratio(best_l1_ratio: object) -> object:
    """Return the selected l1_ratio assigned to the refit model."""
    return best_l1_ratio


@register_atom(witness_cd_cv_refit_copy_x)
@icontract.require(lambda copy_x: _bool(copy_x), "copy_x must be boolean")
@icontract.ensure(
    lambda result, copy_x: _bool(result) and result == copy_x,
    "model copy_X assignment must preserve copy_x",
)
def cd_cv_refit_copy_x(copy_x: bool) -> bool:
    """Return the copy_X value assigned to the refit model."""
    return copy_x


@register_atom(witness_cd_cv_refit_precompute_auto_guard_required)
@icontract.ensure(
    lambda result, precompute: _bool(result)
    and result == (isinstance(precompute, str) and precompute == "auto"),
    "precompute auto guard must match sklearn branching",
)
def cd_cv_refit_precompute_auto_guard_required(precompute: object) -> bool:
    """Return whether LinearModelCV.fit forces model.precompute to False."""
    return isinstance(precompute, str) and precompute == "auto"


@register_atom(witness_cd_cv_refit_precompute_value)
@icontract.require(
    lambda precompute_auto_guard_required: _bool(precompute_auto_guard_required),
    "precompute_auto_guard_required must be boolean",
)
@icontract.ensure(
    lambda result, precompute, precompute_auto_guard_required: (
        result is False if precompute_auto_guard_required else result == precompute
    ),
    "resolved model.precompute must follow the sklearn auto override",
)
def cd_cv_refit_precompute_value(
    precompute: object, precompute_auto_guard_required: bool
) -> object:
    """Return the resolved precompute value assigned to the refit model."""
    return False if precompute_auto_guard_required else precompute


@register_atom(witness_cd_cv_refit_fit_call_uses_sample_weight)
@icontract.ensure(
    lambda result, sample_weight: _bool(result) and result == (sample_weight is not None),
    "fit-dispatch branch must match sample_weight is not None",
)
def cd_cv_refit_fit_call_uses_sample_weight(sample_weight: object) -> bool:
    """Return whether LinearModelCV.fit refits the selected model with sample_weight."""
    return sample_weight is not None
