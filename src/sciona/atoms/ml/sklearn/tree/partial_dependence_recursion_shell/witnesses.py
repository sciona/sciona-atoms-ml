"""Ghost witnesses for sklearn tree partial-dependence recursion atoms."""

from __future__ import annotations


def witness_tree_partial_dependence_grid(grid: object) -> object:
    """Describe the `np.asarray(grid, dtype=DTYPE, order="C")` shell."""
    return grid


def witness_tree_partial_dependence_averaged_predictions(grid: object) -> object:
    """Describe the `np.zeros(shape=grid.shape[0], dtype=np.float64, order="C")` shell."""
    return grid


def witness_tree_partial_dependence_target_features(target_features: object) -> object:
    """Describe the `np.asarray(target_features, dtype=np.intp, order="C")` shell."""
    return target_features


def witness_tree_partial_dependence_result(averaged_predictions: object) -> object:
    """Describe the final `return averaged_predictions` shell."""
    return averaged_predictions
