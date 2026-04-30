"""BIRCH leaf-link rewiring helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .state_models import BirchLeafLinkPlan
from .witnesses import (
    witness_birch_split_leaf_link_plan,
    witness_birch_split_next_neighbor_update_required,
    witness_birch_split_prev_neighbor_update_required,
)


def _bool_value(value: object) -> bool:
    return isinstance(value, bool)


def _plan_valid(value: object, has_prev_leaf: bool, has_next_leaf: bool) -> bool:
    if not isinstance(value, BirchLeafLinkPlan):
        return False
    return bool(
        value.left_prev_role == ("prev" if has_prev_leaf else None)
        and value.left_next_role == "right"
        and value.right_prev_role == "left"
        and value.right_next_role == ("next" if has_next_leaf else None)
        and value.prev_next_role == ("left" if has_prev_leaf else None)
        and value.next_prev_role == ("right" if has_next_leaf else None)
    )


@register_atom(witness_birch_split_prev_neighbor_update_required)
@icontract.require(lambda has_prev_leaf: _bool_value(has_prev_leaf), "has_prev_leaf must be boolean")
@icontract.ensure(lambda result, has_prev_leaf: result == has_prev_leaf, "result must match previous-neighbor presence")
def birch_split_prev_neighbor_update_required(has_prev_leaf: bool) -> bool:
    """Return whether sklearn rewires the previous leaf during a BIRCH split."""
    return bool(has_prev_leaf)


@register_atom(witness_birch_split_next_neighbor_update_required)
@icontract.require(lambda has_next_leaf: _bool_value(has_next_leaf), "has_next_leaf must be boolean")
@icontract.ensure(lambda result, has_next_leaf: result == has_next_leaf, "result must match next-neighbor presence")
def birch_split_next_neighbor_update_required(has_next_leaf: bool) -> bool:
    """Return whether sklearn rewires the next leaf during a BIRCH split."""
    return bool(has_next_leaf)


@register_atom(witness_birch_split_leaf_link_plan)
@icontract.require(lambda has_prev_leaf: _bool_value(has_prev_leaf), "has_prev_leaf must be boolean")
@icontract.require(lambda has_next_leaf: _bool_value(has_next_leaf), "has_next_leaf must be boolean")
@icontract.ensure(lambda result, has_prev_leaf, has_next_leaf: _plan_valid(result, has_prev_leaf, has_next_leaf), "plan must match sklearn's leaf-link rewiring roles")
def birch_split_leaf_link_plan(
    has_prev_leaf: bool,
    has_next_leaf: bool,
) -> BirchLeafLinkPlan:
    """Build the role-level leaf-link rewiring plan used by sklearn's `_split_node`."""
    return BirchLeafLinkPlan(
        left_prev_role="prev" if has_prev_leaf else None,
        left_next_role="right",
        right_prev_role="left",
        right_next_role="next" if has_next_leaf else None,
        prev_next_role="left" if has_prev_leaf else None,
        next_prev_role="right" if has_next_leaf else None,
    )
