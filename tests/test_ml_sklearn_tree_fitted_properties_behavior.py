from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def test_tree_fitted_properties_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.fitted_properties import (
        tree_get_depth_result,
        tree_get_n_leaves_result,
    )

    assert callable(tree_get_depth_result)
    assert callable(tree_get_n_leaves_result)


def test_tree_fitted_properties_match_sklearn_runtime() -> None:
    from sciona.atoms.ml.sklearn.tree.fitted_properties import (
        tree_get_depth_result,
        tree_get_n_leaves_result,
    )

    X, y = load_iris(return_X_y=True)
    est = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y)

    assert tree_get_depth_result(est.tree_.max_depth) == est.get_depth()
    assert tree_get_n_leaves_result(est.tree_.n_leaves) == est.get_n_leaves()


def test_tree_fitted_properties_cover_small_tree_values() -> None:
    from sciona.atoms.ml.sklearn.tree.fitted_properties import (
        tree_get_depth_result,
        tree_get_n_leaves_result,
    )

    assert tree_get_depth_result(0) == 0
    assert tree_get_n_leaves_result(1) == 1


def test_tree_fitted_properties_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.fitted_properties import (
        tree_get_depth_result,
        tree_get_n_leaves_result,
    )

    with pytest.raises(ViolationError):
        tree_get_depth_result(-1)

    with pytest.raises(ViolationError):
        tree_get_n_leaves_result(0)

