"""Ghost witnesses for sklearn tree fitted-property atoms."""

from __future__ import annotations


def witness_tree_get_depth_result(tree_max_depth: int) -> int:
    """Describe BaseDecisionTree.get_depth from a fitted tree max_depth value."""
    return tree_max_depth


def witness_tree_get_n_leaves_result(tree_n_leaves: int) -> int:
    """Describe BaseDecisionTree.get_n_leaves from a fitted tree n_leaves value."""
    return tree_n_leaves

