"""Partial-dependence auto-recursion support shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_gradient_boosting_recursion_supported,
    witness_partial_dependence_recursion_supported_estimator,
    witness_partial_dependence_tree_recursion_supported,
)


def _flag(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_partial_dependence_gradient_boosting_recursion_supported)
@icontract.require(lambda is_base_gradient_boosting: _flag(is_base_gradient_boosting), "is_base_gradient_boosting must be boolean")
@icontract.require(lambda init_is_none: _flag(init_is_none), "init_is_none must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_gradient_boosting_recursion_supported(
    *,
    is_base_gradient_boosting: bool,
    init_is_none: bool,
) -> bool:
    """Decide whether sklearn's BaseGradientBoosting branch can use recursion."""
    return bool(is_base_gradient_boosting and init_is_none)


@register_atom(witness_partial_dependence_tree_recursion_supported)
@icontract.require(lambda is_base_hist_gradient_boosting: _flag(is_base_hist_gradient_boosting), "is_base_hist_gradient_boosting must be boolean")
@icontract.require(lambda is_decision_tree_regressor: _flag(is_decision_tree_regressor), "is_decision_tree_regressor must be boolean")
@icontract.require(lambda is_random_forest_regressor: _flag(is_random_forest_regressor), "is_random_forest_regressor must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_tree_recursion_supported(
    *,
    is_base_hist_gradient_boosting: bool,
    is_decision_tree_regressor: bool,
    is_random_forest_regressor: bool,
) -> bool:
    """Decide whether sklearn's tree-family recursion branch is taken."""
    return bool(
        is_base_hist_gradient_boosting
        or is_decision_tree_regressor
        or is_random_forest_regressor
    )


@register_atom(witness_partial_dependence_recursion_supported_estimator)
@icontract.require(lambda gradient_boosting_supported: _flag(gradient_boosting_supported), "gradient_boosting_supported must be boolean")
@icontract.require(lambda tree_supported: _flag(tree_supported), "tree_supported must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_recursion_supported_estimator(
    *,
    gradient_boosting_supported: bool,
    tree_supported: bool,
) -> bool:
    """Combine sklearn's recursion-eligible estimator-family branches."""
    return bool(gradient_boosting_supported or tree_supported)
