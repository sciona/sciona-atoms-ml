"""Ghost witnesses for sklearn tree pruning-path atoms."""

from __future__ import annotations


def witness_tree_pruning_path_estimator(estimator: object) -> object:
    """Describe the `clone(self).set_params(ccp_alpha=0.0)` shell."""
    return estimator


def witness_tree_pruning_path_result(pruning_path: object) -> object:
    """Describe the final `Bunch(**ccp_pruning_path(est.tree_))` shell."""
    return pruning_path
