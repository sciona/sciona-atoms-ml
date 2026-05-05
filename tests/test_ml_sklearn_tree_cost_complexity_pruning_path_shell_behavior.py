from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import ccp_pruning_path


def test_tree_cost_complexity_pruning_path_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.cost_complexity_pruning_path_shell import (
        tree_pruning_path_estimator,
        tree_pruning_path_result,
    )

    assert callable(tree_pruning_path_estimator)
    assert callable(tree_pruning_path_result)


def test_tree_cost_complexity_pruning_path_shell_matches_sklearn_runtime() -> None:
    from sciona.atoms.ml.sklearn.tree.cost_complexity_pruning_path_shell import (
        tree_pruning_path_estimator,
        tree_pruning_path_result,
    )

    X, y = load_iris(return_X_y=True)
    estimator = DecisionTreeClassifier(random_state=0, max_depth=3, ccp_alpha=0.123)

    cloned = tree_pruning_path_estimator(estimator)
    assert cloned is not estimator
    assert cloned.get_params()["ccp_alpha"] == 0.0
    assert estimator.get_params()["ccp_alpha"] == 0.123

    cloned.fit(X, y)
    pruning_mapping = ccp_pruning_path(cloned.tree_)
    actual = tree_pruning_path_result(pruning_mapping)
    expected = estimator.cost_complexity_pruning_path(X, y)
    assert np.allclose(actual.ccp_alphas, expected.ccp_alphas)
    assert np.allclose(actual.impurities, expected.impurities)


def test_tree_cost_complexity_pruning_path_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.cost_complexity_pruning_path_shell import (
        tree_pruning_path_estimator,
        tree_pruning_path_result,
    )

    with pytest.raises(ViolationError):
        tree_pruning_path_estimator(object())

    with pytest.raises(ViolationError):
        tree_pruning_path_result({"ccp_alphas": np.array([0.0]), "bad": np.array([0.0])})
