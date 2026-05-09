"""Sklearn ElasticNet dense decision callback shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_elastic_net_check_is_fitted_args,
    witness_cd_elastic_net_dense_decision_required,
    witness_cd_elastic_net_dense_super_decision_args,
    witness_cd_elastic_net_dense_super_decision_result,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_cd_elastic_net_check_is_fitted_args)
@icontract.require(lambda estimator: estimator is not None, "estimator must be provided")
@icontract.ensure(
    lambda result, estimator: isinstance(result, tuple)
    and len(result) == 1
    and result[0] is estimator,
    "check_is_fitted args must preserve self identity",
)
def cd_elastic_net_check_is_fitted_args(estimator: object) -> tuple[object]:
    """Return the positional payload for ElasticNet._decision_function check_is_fitted."""
    return (estimator,)


@register_atom(witness_cd_elastic_net_dense_decision_required)
@icontract.require(lambda is_sparse: _bool(is_sparse), "is_sparse must be boolean")
@icontract.ensure(
    lambda result, is_sparse: _bool(result) and result == (not is_sparse),
    "dense decision branch must match not sparse.issparse(X)",
)
def cd_elastic_net_dense_decision_required(is_sparse: bool) -> bool:
    """Return whether ElasticNet._decision_function uses the dense superclass branch."""
    return not is_sparse


@register_atom(witness_cd_elastic_net_dense_super_decision_args)
@icontract.require(lambda estimator: estimator is not None, "estimator must be provided")
@icontract.require(lambda X: X is not None, "X must be provided")
@icontract.ensure(
    lambda result, estimator, X: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is estimator
    and result[1] is X,
    "dense superclass decision args must preserve self and X identity",
)
def cd_elastic_net_dense_super_decision_args(
    estimator: object, X: object
) -> tuple[object, object]:
    """Return the conceptual payload for super()._decision_function(X)."""
    return estimator, X


@register_atom(witness_cd_elastic_net_dense_super_decision_result)
@icontract.require(lambda decision_result: decision_result is not None, "decision_result must be provided")
@icontract.ensure(
    lambda result, decision_result: result is decision_result,
    "dense superclass decision result must pass through unchanged",
)
def cd_elastic_net_dense_super_decision_result(decision_result: object) -> object:
    """Return the dense superclass _decision_function callback result."""
    return decision_result
