from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def test_tree_feature_importances_shell_atom_import() -> None:
    from sciona.atoms.ml.sklearn.tree.feature_importances_shell import tree_feature_importances_result

    assert callable(tree_feature_importances_result)


def test_tree_feature_importances_shell_matches_sklearn_runtime() -> None:
    from sciona.atoms.ml.sklearn.tree.feature_importances_shell import tree_feature_importances_result

    X, y = load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y)

    importances = clf.tree_.compute_feature_importances()
    assert np.allclose(tree_feature_importances_result(importances), clf.feature_importances_)


def test_tree_feature_importances_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.feature_importances_shell import tree_feature_importances_result

    with pytest.raises(ViolationError):
        tree_feature_importances_result(np.array([0.2, -0.1, 0.9], dtype=np.float64))
