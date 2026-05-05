"""Ghost witnesses for sklearn tree fit-API atoms."""

from __future__ import annotations


def witness_tree_classifier_fit_return_self(estimator: object) -> object:
    """Describe the final `return self` shell in DecisionTreeClassifier.fit."""
    return estimator


def witness_tree_regressor_fit_return_self(estimator: object) -> object:
    """Describe the final `return self` shell in DecisionTreeRegressor.fit."""
    return estimator
