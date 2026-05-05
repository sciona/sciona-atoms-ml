"""Sklearn tree estimator-tag atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_decision_tree_classifier_allow_nan_tag,
    witness_decision_tree_regressor_allow_nan_tag,
    witness_extra_tree_classifier_allow_nan_tag,
    witness_extra_tree_regressor_allow_nan_tag,
    witness_tree_classifier_multilabel_tag,
    witness_tree_sparse_input_tag,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _nonempty_str(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


@register_atom(witness_tree_sparse_input_tag)
@icontract.require(lambda parent_sparse=False: _bool(parent_sparse), "parent_sparse must be boolean")
@icontract.ensure(lambda result: _bool(result) and result is True, "tree sparse-input tag must be True")
def tree_sparse_input_tag(parent_sparse: bool = False) -> bool:
    """Return BaseDecisionTree's sparse-input tag override."""
    del parent_sparse
    return True


@register_atom(witness_tree_classifier_multilabel_tag)
@icontract.require(
    lambda parent_multi_label=False: _bool(parent_multi_label),
    "parent_multi_label must be boolean",
)
@icontract.ensure(
    lambda result: _bool(result) and result is True,
    "tree classifier multi_label tag must be True",
)
def tree_classifier_multilabel_tag(parent_multi_label: bool = False) -> bool:
    """Return the shared multi-label tag override for classifier tree estimators."""
    del parent_multi_label
    return True


@register_atom(witness_decision_tree_classifier_allow_nan_tag)
@icontract.require(lambda splitter: _nonempty_str(splitter), "splitter must be a nonempty string")
@icontract.require(lambda criterion: _nonempty_str(criterion), "criterion must be a nonempty string")
@icontract.ensure(lambda result: _bool(result), "allow_nan tag must be boolean")
def decision_tree_classifier_allow_nan_tag(splitter: str, criterion: str) -> bool:
    """Return DecisionTreeClassifier's allow_nan tag rule."""
    return splitter in {"best", "random"} and criterion in {"gini", "log_loss", "entropy"}


@register_atom(witness_decision_tree_regressor_allow_nan_tag)
@icontract.require(lambda splitter: _nonempty_str(splitter), "splitter must be a nonempty string")
@icontract.require(lambda criterion: _nonempty_str(criterion), "criterion must be a nonempty string")
@icontract.ensure(lambda result: _bool(result), "allow_nan tag must be boolean")
def decision_tree_regressor_allow_nan_tag(splitter: str, criterion: str) -> bool:
    """Return DecisionTreeRegressor's allow_nan tag rule."""
    return splitter in {"best", "random"} and criterion in {
        "squared_error",
        "friedman_mse",
        "poisson",
    }


@register_atom(witness_extra_tree_classifier_allow_nan_tag)
@icontract.require(lambda splitter: _nonempty_str(splitter), "splitter must be a nonempty string")
@icontract.require(lambda criterion: _nonempty_str(criterion), "criterion must be a nonempty string")
@icontract.ensure(lambda result: _bool(result), "allow_nan tag must be boolean")
def extra_tree_classifier_allow_nan_tag(splitter: str, criterion: str) -> bool:
    """Return ExtraTreeClassifier's allow_nan tag rule."""
    return splitter == "random" and criterion in {"gini", "log_loss", "entropy"}


@register_atom(witness_extra_tree_regressor_allow_nan_tag)
@icontract.require(lambda parent_allow_nan: _bool(parent_allow_nan), "parent_allow_nan must be boolean")
@icontract.ensure(lambda result: _bool(result), "allow_nan tag must be boolean")
def extra_tree_regressor_allow_nan_tag(parent_allow_nan: bool) -> bool:
    """Return ExtraTreeRegressor's effective allow_nan tag value."""
    return parent_allow_nan

