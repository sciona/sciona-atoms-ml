"""Sklearn tree fit-API atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_classifier_fit_return_self,
    witness_tree_regressor_fit_return_self,
)

def _fitted_classifier(estimator: object) -> bool:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    return isinstance(estimator, DecisionTreeClassifier) and hasattr(estimator, "tree_")

def _fitted_regressor(estimator: object) -> bool:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    return isinstance(estimator, DecisionTreeRegressor) and hasattr(estimator, "tree_")

@register_atom(witness_tree_classifier_fit_return_self)
@icontract.require(
    lambda estimator: _fitted_classifier(estimator),
    "estimator must be a fitted DecisionTreeClassifier",
)
@icontract.ensure(
    lambda result, estimator: result is estimator and _fitted_classifier(result),
    "classifier fit shell must return the same fitted estimator object",
)
def tree_classifier_fit_return_self(
    estimator: DecisionTreeClassifier,
) -> DecisionTreeClassifier:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    """Return the fitted DecisionTreeClassifier from the public fit shell."""
    return estimator

@register_atom(witness_tree_regressor_fit_return_self)
@icontract.require(
    lambda estimator: _fitted_regressor(estimator),
    "estimator must be a fitted DecisionTreeRegressor",
)
@icontract.ensure(
    lambda result, estimator: result is estimator and _fitted_regressor(result),
    "regressor fit shell must return the same fitted estimator object",
)
def tree_regressor_fit_return_self(
    estimator: DecisionTreeRegressor,
) -> DecisionTreeRegressor:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    """Return the fitted DecisionTreeRegressor from the public fit shell."""
    return estimator
