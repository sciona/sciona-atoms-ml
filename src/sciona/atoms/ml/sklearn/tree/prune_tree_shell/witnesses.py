"""Ghost witnesses for sklearn tree pruning helper atoms."""

from __future__ import annotations


def witness_tree_prune_required(ccp_alpha: object) -> object:
    """Describe the `if self.ccp_alpha == 0.0: return` gate in _prune_tree."""
    return ccp_alpha


def witness_tree_prune_classifier_n_classes(n_classes: object) -> object:
    """Describe the `np.atleast_1d(self.n_classes_)` shell in classifier pruning."""
    return n_classes


def witness_tree_prune_regressor_n_classes(n_outputs: object) -> object:
    """Describe the `np.array([1] * self.n_outputs_, dtype=np.intp)` shell in regressor pruning."""
    return n_outputs


def witness_tree_pruned_tree_result(pruned_tree: object) -> object:
    """Describe the final `self.tree_ = pruned_tree` shell in _prune_tree."""
    return pruned_tree
