"""Ghost witnesses for sklearn tree post-build classifier-state atoms."""

from __future__ import annotations


def witness_tree_fit_single_output_classifier_branch(
    n_outputs: object, is_classifier: object
) -> object:
    """Describe the `if self.n_outputs_ == 1 and is_classifier(self)` branch in BaseDecisionTree._fit."""
    return n_outputs, is_classifier


def witness_tree_fit_single_output_n_classes(n_classes: object) -> object:
    """Describe the `self.n_classes_ = self.n_classes_[0]` shell in BaseDecisionTree._fit."""
    return n_classes


def witness_tree_fit_single_output_classes(classes: object) -> object:
    """Describe the `self.classes_ = self.classes_[0]` shell in BaseDecisionTree._fit."""
    return classes
