"""Ghost witnesses for partial-dependence auto-recursion support shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_gradient_boosting_recursion_supported(
    *,
    is_base_gradient_boosting: bool,
    init_is_none: bool,
) -> AbstractArray:
    """Describe sklearn's BaseGradientBoosting recursion-eligibility predicate."""
    del is_base_gradient_boosting, init_is_none
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_tree_recursion_supported(
    *,
    is_base_hist_gradient_boosting: bool,
    is_decision_tree_regressor: bool,
    is_random_forest_regressor: bool,
) -> AbstractArray:
    """Describe sklearn's tree-family recursion-eligibility predicate."""
    del (
        is_base_hist_gradient_boosting,
        is_decision_tree_regressor,
        is_random_forest_regressor,
    )
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_recursion_supported_estimator(
    *,
    gradient_boosting_supported: bool,
    tree_supported: bool,
) -> AbstractArray:
    """Describe the combined recursion-eligibility predicate."""
    del gradient_boosting_supported, tree_supported
    return AbstractArray(shape=(), dtype="bool")
