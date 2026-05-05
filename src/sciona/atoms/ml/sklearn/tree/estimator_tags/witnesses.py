"""Ghost witnesses for sklearn tree estimator-tag atoms."""

from __future__ import annotations


def witness_tree_sparse_input_tag(parent_sparse: bool = False) -> bool:
    """Describe BaseDecisionTree's sparse-input tag override."""
    del parent_sparse
    return True


def witness_tree_classifier_multilabel_tag(parent_multi_label: bool = False) -> bool:
    """Describe the classifier-tree multi-label tag override."""
    del parent_multi_label
    return True


def witness_decision_tree_classifier_allow_nan_tag(splitter: str, criterion: str) -> bool:
    """Describe DecisionTreeClassifier's allow_nan tag rule."""
    return splitter in {"best", "random"} and criterion in {"gini", "log_loss", "entropy"}


def witness_decision_tree_regressor_allow_nan_tag(splitter: str, criterion: str) -> bool:
    """Describe DecisionTreeRegressor's allow_nan tag rule."""
    return splitter in {"best", "random"} and criterion in {"squared_error", "friedman_mse", "poisson"}


def witness_extra_tree_classifier_allow_nan_tag(splitter: str, criterion: str) -> bool:
    """Describe ExtraTreeClassifier's allow_nan tag rule."""
    return splitter == "random" and criterion in {"gini", "log_loss", "entropy"}


def witness_extra_tree_regressor_allow_nan_tag(parent_allow_nan: bool) -> bool:
    """Describe ExtraTreeRegressor's effective allow_nan tag behavior."""
    return parent_allow_nan

