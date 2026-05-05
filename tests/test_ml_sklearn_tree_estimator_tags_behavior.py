from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)


def test_tree_estimator_tags_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.estimator_tags import (
        decision_tree_classifier_allow_nan_tag,
        decision_tree_regressor_allow_nan_tag,
        extra_tree_classifier_allow_nan_tag,
        extra_tree_regressor_allow_nan_tag,
        tree_classifier_multilabel_tag,
        tree_sparse_input_tag,
    )

    assert callable(tree_sparse_input_tag)
    assert callable(tree_classifier_multilabel_tag)
    assert callable(decision_tree_classifier_allow_nan_tag)
    assert callable(decision_tree_regressor_allow_nan_tag)
    assert callable(extra_tree_classifier_allow_nan_tag)
    assert callable(extra_tree_regressor_allow_nan_tag)


def test_tree_estimator_tags_match_runtime_sklearn_behavior() -> None:
    from sciona.atoms.ml.sklearn.tree.estimator_tags import (
        decision_tree_classifier_allow_nan_tag,
        decision_tree_regressor_allow_nan_tag,
        extra_tree_classifier_allow_nan_tag,
        extra_tree_regressor_allow_nan_tag,
        tree_classifier_multilabel_tag,
        tree_sparse_input_tag,
    )

    assert tree_sparse_input_tag(False) is True
    assert tree_classifier_multilabel_tag(False) is True

    dtc = DecisionTreeClassifier(splitter="best", criterion="gini").__sklearn_tags__()
    assert decision_tree_classifier_allow_nan_tag("best", "gini") is dtc.input_tags.allow_nan

    dtr = DecisionTreeRegressor(splitter="best", criterion="squared_error").__sklearn_tags__()
    assert decision_tree_regressor_allow_nan_tag("best", "squared_error") is dtr.input_tags.allow_nan

    etc_best = ExtraTreeClassifier(splitter="best", criterion="gini").__sklearn_tags__()
    assert extra_tree_classifier_allow_nan_tag("best", "gini") is etc_best.input_tags.allow_nan

    etc_random = ExtraTreeClassifier(splitter="random", criterion="gini").__sklearn_tags__()
    assert extra_tree_classifier_allow_nan_tag("random", "gini") is etc_random.input_tags.allow_nan

    etr = ExtraTreeRegressor(splitter="best", criterion="squared_error").__sklearn_tags__()
    assert extra_tree_regressor_allow_nan_tag(True) is etr.input_tags.allow_nan


def test_tree_estimator_tags_cover_false_cases_and_passthrough() -> None:
    from sciona.atoms.ml.sklearn.tree.estimator_tags import (
        decision_tree_classifier_allow_nan_tag,
        decision_tree_regressor_allow_nan_tag,
        extra_tree_classifier_allow_nan_tag,
        extra_tree_regressor_allow_nan_tag,
    )

    assert decision_tree_classifier_allow_nan_tag("best", "absolute_error") is False
    assert decision_tree_regressor_allow_nan_tag("best", "absolute_error") is False
    assert extra_tree_classifier_allow_nan_tag("best", "gini") is False
    assert extra_tree_regressor_allow_nan_tag(False) is False

    etr = ExtraTreeRegressor(splitter="best", criterion="absolute_error").__sklearn_tags__()
    assert extra_tree_regressor_allow_nan_tag(False) is etr.input_tags.allow_nan


def test_tree_estimator_tags_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.estimator_tags import (
        decision_tree_classifier_allow_nan_tag,
        extra_tree_regressor_allow_nan_tag,
        tree_sparse_input_tag,
    )

    with pytest.raises(ViolationError):
        tree_sparse_input_tag(1)

    with pytest.raises(ViolationError):
        decision_tree_classifier_allow_nan_tag("", "gini")

    with pytest.raises(ViolationError):
        extra_tree_regressor_allow_nan_tag(1)

