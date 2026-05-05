"""Ghost witnesses for sklearn tree prediction-output atoms."""

from __future__ import annotations


def witness_tree_classifier_single_output_labels(
    probabilities: object,
    classes: object,
) -> object:
    """Describe single-output classifier label decoding from tree_.predict(X)."""
    del probabilities
    return classes


def witness_tree_classifier_multioutput_labels(
    probabilities: object,
    classes_blocks: object,
) -> object:
    """Describe multioutput classifier label decoding from tree_.predict(X)."""
    del probabilities
    return classes_blocks


def witness_tree_regressor_single_output_values(probabilities: object) -> object:
    """Describe single-output regressor value selection from tree_.predict(X)."""
    return probabilities


def witness_tree_regressor_multioutput_values(probabilities: object) -> object:
    """Describe multioutput regressor value selection from tree_.predict(X)."""
    return probabilities

