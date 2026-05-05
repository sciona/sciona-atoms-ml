from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def test_tree_fit_postbuild_classifier_state_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.fit_postbuild_classifier_state import (
        tree_fit_single_output_classes,
        tree_fit_single_output_classifier_branch,
        tree_fit_single_output_n_classes,
    )

    assert callable(tree_fit_single_output_classifier_branch)
    assert callable(tree_fit_single_output_n_classes)
    assert callable(tree_fit_single_output_classes)


def test_tree_fit_postbuild_classifier_state_matches_sklearn_runtime() -> None:
    from sciona.atoms.ml.sklearn.tree.fit_postbuild_classifier_state import (
        tree_fit_single_output_classes,
        tree_fit_single_output_classifier_branch,
        tree_fit_single_output_n_classes,
    )

    X, y = load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y)

    classes_blocks = np.asarray([clf.classes_], dtype=object)
    n_classes_blocks = np.asarray([clf.n_classes_], dtype=np.intp)

    assert tree_fit_single_output_classifier_branch(clf.n_outputs_, True) is True
    assert tree_fit_single_output_n_classes(n_classes_blocks) == clf.n_classes_
    assert np.array_equal(tree_fit_single_output_classes(classes_blocks), clf.classes_)


def test_tree_fit_postbuild_classifier_state_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.fit_postbuild_classifier_state import (
        tree_fit_single_output_classes,
        tree_fit_single_output_classifier_branch,
        tree_fit_single_output_n_classes,
    )

    with pytest.raises(ViolationError):
        tree_fit_single_output_classifier_branch(0, True)

    with pytest.raises(ViolationError):
        tree_fit_single_output_n_classes(np.array([], dtype=np.intp))

    with pytest.raises(ViolationError):
        tree_fit_single_output_classes(np.array(["a", "b"], dtype=object))
