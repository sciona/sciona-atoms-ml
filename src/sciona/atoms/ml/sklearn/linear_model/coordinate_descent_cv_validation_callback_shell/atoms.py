"""Sklearn coordinate-descent CV validation callback-shell atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_check_consistent_length_args,
    witness_cd_cv_validate_data_args,
    witness_cd_cv_validate_data_kwargs,
    witness_cd_cv_validated_x,
    witness_cd_cv_validated_y,
)


def _validated_pair(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2


@register_atom(witness_cd_cv_validate_data_args)
@icontract.ensure(
    lambda result, estimator, X, y: isinstance(result, tuple)
    and len(result) == 3
    and result[0] is estimator
    and result[1] is X
    and result[2] is y,
    "validate_data positional args must preserve estimator, X, and y identity",
)
def cd_cv_validate_data_args(estimator: object, X: object, y: object) -> tuple[object, object, object]:
    """Return positional args for validate_data(self, X, y, ...)."""
    return (estimator, X, y)


@register_atom(witness_cd_cv_validate_data_kwargs)
@icontract.require(lambda check_x_params: isinstance(check_x_params, dict), "check_x_params must be a dict")
@icontract.require(lambda check_y_params: isinstance(check_y_params, dict), "check_y_params must be a dict")
@icontract.ensure(
    lambda result, check_x_params, check_y_params: isinstance(result, dict)
    and set(result) == {"validate_separately"}
    and result["validate_separately"][0] == check_x_params
    and result["validate_separately"][1] == check_y_params,
    "validate_data kwargs must package validate_separately=(check_X_params, check_y_params)",
)
def cd_cv_validate_data_kwargs(
    check_x_params: dict[object, object], check_y_params: dict[object, object]
) -> dict[str, tuple[dict[object, object], dict[object, object]]]:
    """Return kwargs for validate_data(..., validate_separately=(...))."""
    return {"validate_separately": (dict(check_x_params), dict(check_y_params))}


@register_atom(witness_cd_cv_validated_x)
@icontract.require(lambda validated_pair: _validated_pair(validated_pair), "validated_pair must be a two-tuple")
@icontract.ensure(
    lambda result, validated_pair: result is validated_pair[0],
    "validated X extraction must preserve tuple element identity",
)
def cd_cv_validated_x(validated_pair: tuple[object, object]) -> object:
    """Return X from the deferred validate_data(...) result tuple."""
    return validated_pair[0]


@register_atom(witness_cd_cv_validated_y)
@icontract.require(lambda validated_pair: _validated_pair(validated_pair), "validated_pair must be a two-tuple")
@icontract.ensure(
    lambda result, validated_pair: result is validated_pair[1],
    "validated y extraction must preserve tuple element identity",
)
def cd_cv_validated_y(validated_pair: tuple[object, object]) -> object:
    """Return y from the deferred validate_data(...) result tuple."""
    return validated_pair[1]


@register_atom(witness_cd_cv_check_consistent_length_args)
@icontract.ensure(
    lambda result, X, y: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is X
    and result[1] is y,
    "check_consistent_length args must preserve X and y identity",
)
def cd_cv_check_consistent_length_args(X: object, y: object) -> tuple[object, object]:
    """Return positional args for check_consistent_length(X, y)."""
    return (X, y)
