"""Ghost witnesses for sklearn tree path-API atoms."""

from __future__ import annotations


def witness_tree_apply_leaf_indices(leaf_indices: object) -> object:
    """Describe the final `return self.tree_.apply(X)` shell in BaseDecisionTree.apply."""
    return leaf_indices


def witness_tree_decision_path_indicator(indicator: object) -> object:
    """Describe the final `return self.tree_.decision_path(X)` shell in BaseDecisionTree.decision_path."""
    return indicator

