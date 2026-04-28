from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import partial_dependence
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeRegressor


class DummyClusterer(ClusterMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "DummyClusterer":
        del X, y
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        del X
        return np.zeros(1, dtype=np.int64)


def test_partial_dependence_task_guard_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_task_guards import (
        partial_dependence_require_classifier_or_regressor,
        partial_dependence_require_decision_function_for_recursion,
        partial_dependence_require_not_multiclass_multioutput,
        partial_dependence_resolve_recursion_response_method,
    )

    assert callable(partial_dependence_require_classifier_or_regressor)
    assert callable(partial_dependence_require_not_multiclass_multioutput)
    assert callable(partial_dependence_resolve_recursion_response_method)
    assert callable(partial_dependence_require_decision_function_for_recursion)


def test_partial_dependence_supported_task_guard_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_task_guards import (
        partial_dependence_require_classifier_or_regressor,
    )

    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    est = DummyClusterer().fit(X)

    assert partial_dependence_require_classifier_or_regressor("classifier") == "classifier"
    assert partial_dependence_require_classifier_or_regressor("regressor") == "regressor"

    with pytest.raises(ValueError, match="fitted regressor or classifier"):
        partial_dependence_require_classifier_or_regressor("other")
    with pytest.raises(ValueError, match="fitted regressor or classifier"):
        partial_dependence(est, X=X, features=[0])


def test_partial_dependence_multiclass_multioutput_guard_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_task_guards import (
        partial_dependence_require_not_multiclass_multioutput,
    )

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 2],
            [1, 0],
            [2, 1],
        ],
        dtype=np.int64,
    )
    est = MultiOutputClassifier(DummyClassifier(strategy="most_frequent")).fit(X, y)

    assert partial_dependence_require_not_multiclass_multioutput(
        is_classifier_task=False,
        classes_are_multioutput=True,
    ) is True

    with pytest.raises(ValueError, match="Multiclass-multioutput estimators are not supported"):
        partial_dependence_require_not_multiclass_multioutput(
            is_classifier_task=True,
            classes_are_multioutput=True,
        )
    with pytest.raises(ValueError, match="Multiclass-multioutput estimators are not supported"):
        partial_dependence(est, X=X, features=[0])


def test_partial_dependence_recursion_response_method_resolution_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_task_guards import (
        partial_dependence_resolve_recursion_response_method,
    )

    assert partial_dependence_resolve_recursion_response_method("auto") == "decision_function"
    assert partial_dependence_resolve_recursion_response_method("predict_proba") == "predict_proba"
    assert partial_dependence_resolve_recursion_response_method("decision_function") == "decision_function"


def test_partial_dependence_recursion_decision_function_guard_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_task_guards import (
        partial_dependence_require_decision_function_for_recursion,
        partial_dependence_resolve_recursion_response_method,
    )

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y_clf = np.array([0, 0, 1, 1], dtype=np.int64)
    clf = GradientBoostingClassifier(random_state=0).fit(X, y_clf)

    assert partial_dependence_require_decision_function_for_recursion("decision_function") == "decision_function"
    assert (
        partial_dependence_require_decision_function_for_recursion(
            partial_dependence_resolve_recursion_response_method("auto")
        )
        == "decision_function"
    )

    with pytest.raises(ValueError, match="response_method must be 'decision_function'"):
        partial_dependence_require_decision_function_for_recursion("predict_proba")
    with pytest.raises(ValueError, match="response_method must be 'decision_function'"):
        partial_dependence(clf, X=X, features=[0], method="recursion", response_method="predict_proba")


def test_partial_dependence_task_guard_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_task_guards import (
        partial_dependence_require_classifier_or_regressor,
        partial_dependence_require_decision_function_for_recursion,
        partial_dependence_require_not_multiclass_multioutput,
        partial_dependence_resolve_recursion_response_method,
    )

    with pytest.raises(Exception):
        partial_dependence_require_classifier_or_regressor("clusterer")
    with pytest.raises(Exception):
        partial_dependence_require_not_multiclass_multioutput(
            is_classifier_task=np.bool_(True),
            classes_are_multioutput=False,
        )
    with pytest.raises(Exception):
        partial_dependence_resolve_recursion_response_method("predict")
    with pytest.raises(Exception):
        partial_dependence_require_decision_function_for_recursion("predict")
