"""Ghost witnesses for BIRCH leaf-link rewiring helpers."""

from __future__ import annotations

from .state_models import BirchLeafLinkPlan


def witness_birch_split_prev_neighbor_update_required(has_prev_leaf: bool) -> bool:
    """Describe whether a previous leaf neighbor must be rewired."""
    return bool(has_prev_leaf)


def witness_birch_split_next_neighbor_update_required(has_next_leaf: bool) -> bool:
    """Describe whether a next leaf neighbor must be rewired."""
    return bool(has_next_leaf)


def witness_birch_split_leaf_link_plan(
    has_prev_leaf: bool,
    has_next_leaf: bool,
) -> BirchLeafLinkPlan:
    """Describe the role-level leaf-link plan after a BIRCH split."""
    return BirchLeafLinkPlan(
        left_prev_role="prev" if has_prev_leaf else None,
        left_next_role="right",
        right_prev_role="left",
        right_next_role="next" if has_next_leaf else None,
        prev_next_role="left" if has_prev_leaf else None,
        next_prev_role="right" if has_next_leaf else None,
    )
