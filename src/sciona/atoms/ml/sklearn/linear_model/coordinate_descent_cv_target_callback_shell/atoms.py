"""Sklearn coordinate-descent CV target and sample-weight callback-shell atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_check_sample_weight_args,
    witness_cd_cv_check_sample_weight_kwargs,
    witness_cd_cv_checked_sample_weight,
    witness_cd_cv_column_or_1d_args,
    witness_cd_cv_column_or_1d_result,
    witness_cd_cv_is_multitask_result,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_cd_cv_is_multitask_result)
@icontract.require(lambda multitask: _bool(multitask), "multitask must be boolean")
@icontract.ensure(
    lambda result, multitask: _bool(result) and result == multitask,
    "_is_multitask callback result must pass through unchanged",
)
def cd_cv_is_multitask_result(multitask: bool) -> bool:
    """Return the boolean result from the deferred self._is_multitask() callback."""
    return multitask


@register_atom(witness_cd_cv_column_or_1d_args)
@icontract.require(lambda multitask: multitask is False, "column_or_1d is used only on the non-multitask branch")
@icontract.ensure(
    lambda result, y: isinstance(result, tuple)
    and len(result) == 1
    and result[0] is y,
    "column_or_1d positional args must preserve y identity",
)
def cd_cv_column_or_1d_args(y: object, multitask: bool) -> tuple[object]:
    """Return positional args for column_or_1d(y, warn=True)."""
    del multitask
    return (y,)


@register_atom(witness_cd_cv_column_or_1d_result)
@icontract.ensure(
    lambda result, normalized_y: result is normalized_y,
    "column_or_1d callback result must preserve normalized y identity",
)
def cd_cv_column_or_1d_result(normalized_y: object) -> object:
    """Return y after the deferred column_or_1d(y, warn=True) callback."""
    return normalized_y


@register_atom(witness_cd_cv_check_sample_weight_args)
@icontract.require(lambda sample_weight: sample_weight is not None, "sample_weight must be present")
@icontract.ensure(
    lambda result, sample_weight, X: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is sample_weight
    and result[1] is X,
    "_check_sample_weight positional args must preserve sample_weight and X identity",
)
def cd_cv_check_sample_weight_args(sample_weight: object, X: object) -> tuple[object, object]:
    """Return positional args for _check_sample_weight(sample_weight, X, dtype=X.dtype)."""
    return (sample_weight, X)


@register_atom(witness_cd_cv_check_sample_weight_kwargs)
@icontract.ensure(
    lambda result, x_dtype: isinstance(result, dict) and result == {"dtype": x_dtype},
    "_check_sample_weight kwargs must map dtype through unchanged",
)
def cd_cv_check_sample_weight_kwargs(x_dtype: object) -> dict[str, object]:
    """Return kwargs for _check_sample_weight(..., dtype=X.dtype)."""
    return {"dtype": x_dtype}


@register_atom(witness_cd_cv_checked_sample_weight)
@icontract.ensure(
    lambda result, checked_sample_weight: result is checked_sample_weight,
    "_check_sample_weight callback result must preserve checked sample_weight identity",
)
def cd_cv_checked_sample_weight(checked_sample_weight: object) -> object:
    """Return sample_weight after the deferred _check_sample_weight(...) callback."""
    return checked_sample_weight
