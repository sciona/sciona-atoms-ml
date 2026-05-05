from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def test_tree_classifier_probability_outputs_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.classifier_probability_outputs import (
        tree_predict_log_proba_multioutput,
        tree_predict_log_proba_single_output,
        tree_predict_proba_multioutput,
        tree_predict_proba_single_output,
    )

    assert callable(tree_predict_proba_single_output)
    assert callable(tree_predict_proba_multioutput)
    assert callable(tree_predict_log_proba_single_output)
    assert callable(tree_predict_log_proba_multioutput)


def test_tree_classifier_probability_outputs_match_sklearn_runtime() -> None:
    from sciona.atoms.ml.sklearn.tree.classifier_probability_outputs import (
        tree_predict_log_proba_multioutput,
        tree_predict_log_proba_single_output,
        tree_predict_proba_multioutput,
        tree_predict_proba_single_output,
    )

    X, y = load_iris(return_X_y=True)
    X_tree = X.astype(np.float32)

    clf = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y)
    raw_single = clf.tree_.predict(X_tree)
    actual_single = tree_predict_proba_single_output(raw_single, int(clf.n_classes_))
    expected_single = clf.predict_proba(X)
    assert np.allclose(actual_single, expected_single)
    with np.errstate(divide="ignore"):
        expected_log_single = clf.predict_log_proba(X)
    assert np.allclose(tree_predict_log_proba_single_output(actual_single), expected_log_single)

    y_multi = np.column_stack([y, y % 2])
    clf_multi = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y_multi)
    raw_multi = clf_multi.tree_.predict(X_tree)
    n_classes = tuple(int(value) for value in np.asarray(clf_multi.n_classes_))
    actual_multi = tree_predict_proba_multioutput(raw_multi, n_classes)
    expected_multi = clf_multi.predict_proba(X)
    assert len(actual_multi) == len(expected_multi)
    for actual_block, expected_block in zip(actual_multi, expected_multi, strict=True):
        assert np.allclose(actual_block, expected_block)

    actual_log_multi = tree_predict_log_proba_multioutput(tuple(actual_multi))
    with np.errstate(divide="ignore"):
        expected_log_multi = clf_multi.predict_log_proba(X)
    for actual_block, expected_block in zip(actual_log_multi, expected_log_multi, strict=True):
        assert np.allclose(actual_block, expected_block)


def test_tree_classifier_probability_outputs_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.classifier_probability_outputs import (
        tree_predict_log_proba_single_output,
        tree_predict_proba_single_output,
    )

    with pytest.raises(ViolationError):
        tree_predict_proba_single_output(np.array([[0.5, 0.5]], dtype=np.float64), 0)

    with pytest.raises(ViolationError):
        tree_predict_log_proba_single_output(np.array([0.5, 0.5], dtype=np.float64))
