"""Sklearn tree prediction-branching atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_predict_sample_count,
    witness_tree_predict_use_classifier_branch,
    witness_tree_predict_use_single_output_branch,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@register_atom(witness_tree_predict_sample_count)
@icontract.require(lambda sample_count: _positive_int(sample_count), "sample_count must be a positive integer")
@icontract.ensure(
    lambda result, sample_count: _positive_int(result) and result == sample_count,
    "predict sample count must preserve the validated input sample count",
)
def tree_predict_sample_count(sample_count: int) -> int:
    """Return BaseDecisionTree.predict's `n_samples = X.shape[0]` bookkeeping value."""
    return sample_count


@register_atom(witness_tree_predict_use_classifier_branch)
@icontract.require(lambda is_classifier_task: isinstance(is_classifier_task, bool), "is_classifier_task must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def tree_predict_use_classifier_branch(is_classifier_task: bool) -> bool:
    """Return the classifier-versus-regressor branch predicate in BaseDecisionTree.predict."""
    return is_classifier_task


@register_atom(witness_tree_predict_use_single_output_branch)
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def tree_predict_use_single_output_branch(n_outputs: int) -> bool:
    """Return the single-output-versus-multioutput branch predicate in BaseDecisionTree.predict."""
    return n_outputs == 1

