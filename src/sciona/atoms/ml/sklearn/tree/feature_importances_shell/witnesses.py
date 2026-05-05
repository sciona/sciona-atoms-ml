"""Ghost witnesses for sklearn tree feature-importance atoms."""

from __future__ import annotations


def witness_tree_feature_importances_result(importances: object) -> object:
    """Describe the final `return self.tree_.compute_feature_importances()` shell."""
    return importances
