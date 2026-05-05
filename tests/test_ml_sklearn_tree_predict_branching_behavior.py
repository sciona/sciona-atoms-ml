from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def test_tree_predict_branching_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.predict_branching import (
        tree_predict_sample_count,
        tree_predict_use_classifier_branch,
        tree_predict_use_single_output_branch,
    )

    assert callable(tree_predict_sample_count)
    assert callable(tree_predict_use_classifier_branch)
    assert callable(tree_predict_use_single_output_branch)


def test_tree_predict_branching_matches_sklearn_runtime() -> None:
    from sciona.atoms.ml.sklearn.tree.predict_branching import (
        tree_predict_sample_count,
        tree_predict_use_classifier_branch,
        tree_predict_use_single_output_branch,
    )

    X, y = load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y)

    assert tree_predict_sample_count(X.shape[0]) == X.shape[0]
    assert tree_predict_use_classifier_branch(True) is True
    assert tree_predict_use_classifier_branch(False) is False
    assert tree_predict_use_single_output_branch(int(clf.n_outputs_)) is True
    assert tree_predict_use_single_output_branch(2) is False


def test_tree_predict_branching_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.predict_branching import (
        tree_predict_sample_count,
        tree_predict_use_classifier_branch,
        tree_predict_use_single_output_branch,
    )

    with pytest.raises(ViolationError):
        tree_predict_sample_count(0)

    with pytest.raises(ViolationError):
        tree_predict_use_classifier_branch(1)

    with pytest.raises(ViolationError):
        tree_predict_use_single_output_branch(0)

