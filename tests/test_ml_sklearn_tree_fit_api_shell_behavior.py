from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def test_tree_fit_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.fit_api_shell import (
        tree_classifier_fit_return_self,
        tree_regressor_fit_return_self,
    )

    assert callable(tree_classifier_fit_return_self)
    assert callable(tree_regressor_fit_return_self)


def test_tree_fit_api_shell_matches_sklearn_runtime() -> None:
    from sciona.atoms.ml.sklearn.tree.fit_api_shell import (
        tree_classifier_fit_return_self,
        tree_regressor_fit_return_self,
    )

    X, y = load_iris(return_X_y=True)

    clf = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y)
    assert tree_classifier_fit_return_self(clf) is clf

    reg = DecisionTreeRegressor(random_state=0, max_depth=3).fit(X, y.astype(np.float64))
    assert tree_regressor_fit_return_self(reg) is reg


def test_tree_fit_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.fit_api_shell import (
        tree_classifier_fit_return_self,
        tree_regressor_fit_return_self,
    )

    with pytest.raises(ViolationError):
        tree_classifier_fit_return_self(DecisionTreeClassifier())

    with pytest.raises(ViolationError):
        tree_regressor_fit_return_self(DecisionTreeRegressor())
