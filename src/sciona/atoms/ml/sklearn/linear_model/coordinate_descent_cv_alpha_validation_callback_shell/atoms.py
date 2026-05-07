"""Sklearn coordinate-descent CV alpha validation callback-shell atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_alpha_check_scalar_args,
    witness_cd_cv_alpha_check_scalar_kwargs,
    witness_cd_cv_alpha_check_scalar_result,
    witness_cd_cv_user_alpha_validation_required,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and value >= 0


@register_atom(witness_cd_cv_user_alpha_validation_required)
@icontract.require(lambda alphas_is_none: _bool(alphas_is_none), "alphas_is_none must be boolean")
@icontract.ensure(
    lambda result, alphas_is_none: _bool(result) and result == (not alphas_is_none),
    "user-alpha validation branch must match alphas is not None",
)
def cd_cv_user_alpha_validation_required(alphas_is_none: bool) -> bool:
    """Return whether LinearModelCV.fit validates user-provided alphas."""
    return not alphas_is_none


@register_atom(witness_cd_cv_alpha_check_scalar_kwargs)
@icontract.require(lambda target_type: target_type is not None, "target_type must be provided")
@icontract.ensure(
    lambda result, target_type: isinstance(result, dict)
    and result == {
        "target_type": target_type,
        "min_val": 0.0,
        "include_boundaries": "left",
    },
    "check_scalar kwargs must match sklearn's user-alpha validator",
)
def cd_cv_alpha_check_scalar_kwargs(target_type: object) -> dict[str, object]:
    """Return kwargs for check_scalar on each user-provided alpha."""
    return {
        "target_type": target_type,
        "min_val": 0.0,
        "include_boundaries": "left",
    }


@register_atom(witness_cd_cv_alpha_check_scalar_args)
@icontract.require(lambda index: _nonnegative_int(index), "index must be nonnegative")
@icontract.ensure(
    lambda result, alpha, index: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is alpha
    and result[1] == f"alphas[{index}]",
    "check_scalar args must preserve alpha identity and format the indexed name",
)
def cd_cv_alpha_check_scalar_args(alpha: object, index: int) -> tuple[object, str]:
    """Return positional args for check_scalar(alpha, f'alphas[{index}]', ...)."""
    return (alpha, f"alphas[{index}]")


@register_atom(witness_cd_cv_alpha_check_scalar_result)
@icontract.ensure(
    lambda result, checked_alpha: result is checked_alpha,
    "check_scalar callback result must preserve checked alpha identity",
)
def cd_cv_alpha_check_scalar_result(checked_alpha: object) -> object:
    """Return the alpha produced by the deferred check_scalar(...) callback."""
    return checked_alpha
