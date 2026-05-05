from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor


def test_tree_partial_dependence_recursion_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.partial_dependence_recursion_shell import (
        tree_partial_dependence_averaged_predictions,
        tree_partial_dependence_grid,
        tree_partial_dependence_result,
        tree_partial_dependence_target_features,
    )

    assert callable(tree_partial_dependence_grid)
    assert callable(tree_partial_dependence_averaged_predictions)
    assert callable(tree_partial_dependence_target_features)
    assert callable(tree_partial_dependence_result)


def test_tree_partial_dependence_recursion_shell_matches_sklearn_runtime() -> None:
    from sciona.atoms.ml.sklearn.tree.partial_dependence_recursion_shell import (
        tree_partial_dependence_averaged_predictions,
        tree_partial_dependence_grid,
        tree_partial_dependence_result,
        tree_partial_dependence_target_features,
    )

    X, y = load_diabetes(return_X_y=True)
    reg = DecisionTreeRegressor(random_state=0, max_depth=3).fit(X, y)

    grid = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    targets = np.array([0, 1], dtype=np.int64)

    actual_grid = tree_partial_dependence_grid(grid)
    actual_targets = tree_partial_dependence_target_features(targets)
    actual_buffer = tree_partial_dependence_averaged_predictions(actual_grid)

    expected = reg._compute_partial_dependence_recursion(grid, targets)
    reg.tree_.compute_partial_dependence(actual_grid, actual_targets, actual_buffer)
    assert np.allclose(tree_partial_dependence_result(actual_buffer), expected)


def test_tree_partial_dependence_recursion_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.partial_dependence_recursion_shell import (
        tree_partial_dependence_grid,
        tree_partial_dependence_target_features,
    )

    with pytest.raises(ViolationError):
        tree_partial_dependence_grid(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        tree_partial_dependence_target_features(np.array([[0, 1]], dtype=np.int64))
