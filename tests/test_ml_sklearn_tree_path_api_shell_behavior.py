from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def test_tree_path_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.path_api_shell import (
        tree_apply_leaf_indices,
        tree_decision_path_indicator,
    )

    assert callable(tree_apply_leaf_indices)
    assert callable(tree_decision_path_indicator)


def test_tree_path_api_shell_matches_sklearn_runtime() -> None:
    from sciona.atoms.ml.sklearn.tree.path_api_shell import (
        tree_apply_leaf_indices,
        tree_decision_path_indicator,
    )

    X, y = load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y)
    X_tree = X.astype(np.float32)

    leaves = clf.tree_.apply(X_tree).astype(np.int64)
    indicator = clf.tree_.decision_path(X_tree)

    assert np.array_equal(tree_apply_leaf_indices(leaves), clf.apply(X))
    actual_indicator = tree_decision_path_indicator(indicator)
    expected_indicator = clf.decision_path(X)
    assert actual_indicator.shape == expected_indicator.shape
    assert np.array_equal(actual_indicator.toarray(), expected_indicator.toarray())


def test_tree_path_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.path_api_shell import (
        tree_apply_leaf_indices,
        tree_decision_path_indicator,
    )

    with pytest.raises(ViolationError):
        tree_apply_leaf_indices(np.array([[0, 1]], dtype=np.int64))

    with pytest.raises(ViolationError):
        tree_decision_path_indicator(np.array([[1.0, 0.0]], dtype=np.float64))

