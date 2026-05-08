"""Sklearn coordinate-descent estimator intercept callback atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_estimator_fit_return_self,
    witness_cd_estimator_set_intercept_args,
)


@register_atom(witness_cd_estimator_set_intercept_args)
@icontract.require(lambda X_offset: X_offset is not None, "X_offset must be provided")
@icontract.require(lambda y_offset: y_offset is not None, "y_offset must be provided")
@icontract.require(lambda X_scale: X_scale is not None, "X_scale must be provided")
@icontract.ensure(
    lambda result, X_offset, y_offset, X_scale: isinstance(result, tuple)
    and len(result) == 3
    and result[0] is X_offset
    and result[1] is y_offset
    and result[2] is X_scale,
    "set-intercept args must match ElasticNet.fit call order",
)
def cd_estimator_set_intercept_args(
    X_offset: object,
    y_offset: object,
    X_scale: object,
) -> tuple[object, object, object]:
    """Return the positional payload for ElasticNet.fit `_set_intercept`."""
    return X_offset, y_offset, X_scale


@register_atom(witness_cd_estimator_fit_return_self)
@icontract.require(lambda estimator_identity: estimator_identity is not None, "estimator_identity must be provided")
@icontract.ensure(
    lambda result, estimator_identity: result is estimator_identity,
    "fit return must pass self through unchanged",
)
def cd_estimator_fit_return_self(estimator_identity: object) -> object:
    """Return self from ElasticNet.fit."""
    return estimator_identity
