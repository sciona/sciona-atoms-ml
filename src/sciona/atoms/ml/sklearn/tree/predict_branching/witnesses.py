"""Ghost witnesses for sklearn tree prediction-branching atoms."""

from __future__ import annotations


def witness_tree_predict_sample_count(sample_count: int) -> int:
    """Describe the `n_samples = X.shape[0]` bookkeeping in BaseDecisionTree.predict."""
    return sample_count


def witness_tree_predict_use_classifier_branch(is_classifier_task: bool) -> bool:
    """Describe the classifier-versus-regressor branch predicate in BaseDecisionTree.predict."""
    return is_classifier_task


def witness_tree_predict_use_single_output_branch(n_outputs: int) -> bool:
    """Describe the `self.n_outputs_ == 1` branch predicate in BaseDecisionTree.predict."""
    return n_outputs == 1

