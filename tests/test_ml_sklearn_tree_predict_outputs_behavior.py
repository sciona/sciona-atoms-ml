from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def test_tree_predict_outputs_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.predict_outputs import (
        tree_classifier_multioutput_labels,
        tree_classifier_single_output_labels,
        tree_regressor_multioutput_values,
        tree_regressor_single_output_values,
    )

    assert callable(tree_classifier_single_output_labels)
    assert callable(tree_classifier_multioutput_labels)
    assert callable(tree_regressor_single_output_values)
    assert callable(tree_regressor_multioutput_values)


def test_tree_predict_outputs_match_sklearn_runtime() -> None:
    from sciona.atoms.ml.sklearn.tree.predict_outputs import (
        tree_classifier_multioutput_labels,
        tree_classifier_single_output_labels,
        tree_regressor_multioutput_values,
        tree_regressor_single_output_values,
    )

    X, y = load_iris(return_X_y=True)
    X_tree = X.astype(np.float32)

    clf = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y)
    proba = clf.tree_.predict(X_tree)
    assert np.array_equal(tree_classifier_single_output_labels(proba, clf.classes_), clf.predict(X))

    y_multi = np.column_stack([y, y % 2])
    clf_multi = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y_multi)
    proba_multi = clf_multi.tree_.predict(X_tree)
    classes_blocks = tuple(np.asarray(block, dtype=object) for block in clf_multi.classes_)
    assert np.array_equal(tree_classifier_multioutput_labels(proba_multi, classes_blocks), clf_multi.predict(X))

    reg = DecisionTreeRegressor(random_state=0, max_depth=3).fit(X, y.astype(np.float64))
    reg_proba = reg.tree_.predict(X_tree)
    assert np.allclose(tree_regressor_single_output_values(reg_proba), reg.predict(X))

    y_reg_multi = np.column_stack([y.astype(np.float64), (y * 0.5).astype(np.float64)])
    reg_multi = DecisionTreeRegressor(random_state=0, max_depth=3).fit(X, y_reg_multi)
    reg_proba_multi = reg_multi.tree_.predict(X_tree)
    assert np.allclose(tree_regressor_multioutput_values(reg_proba_multi), reg_multi.predict(X))


def test_tree_predict_outputs_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.predict_outputs import (
        tree_classifier_single_output_labels,
        tree_regressor_single_output_values,
    )

    with pytest.raises(ViolationError):
        tree_classifier_single_output_labels(
            np.array([[0.5, 0.5]], dtype=np.float64),
            np.array([[0, 1]], dtype=object),
        )

    with pytest.raises(ViolationError):
        tree_regressor_single_output_values(np.array([1.0, 2.0], dtype=np.float64))
