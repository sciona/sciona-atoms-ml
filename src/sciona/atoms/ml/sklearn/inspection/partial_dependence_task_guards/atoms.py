"""Partial-dependence task and recursion-response guard helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_require_classifier_or_regressor,
    witness_partial_dependence_require_decision_function_for_recursion,
    witness_partial_dependence_require_not_multiclass_multioutput,
    witness_partial_dependence_resolve_recursion_response_method,
)

VALID_TASK_KINDS = {"classifier", "regressor", "other"}
VALID_RESPONSE_METHODS = {"auto", "predict_proba", "decision_function"}


def _task_kind_valid(task_kind: object) -> bool:
    return isinstance(task_kind, str) and task_kind in VALID_TASK_KINDS


def _flag_valid(value: object) -> bool:
    return isinstance(value, bool)


def _response_method_valid(response_method: object) -> bool:
    return isinstance(response_method, str) and response_method in VALID_RESPONSE_METHODS


@register_atom(witness_partial_dependence_require_classifier_or_regressor)
@icontract.require(lambda task_kind: _task_kind_valid(task_kind), "task_kind must be 'classifier', 'regressor', or 'other'")
@icontract.ensure(lambda result: result in {"classifier", "regressor"}, "result must be a supported estimator task kind")
def partial_dependence_require_classifier_or_regressor(task_kind: str) -> str:
    """Reject unsupported estimator task kinds before partial-dependence execution."""
    if task_kind not in {"classifier", "regressor"}:
        raise ValueError("'estimator' must be a fitted regressor or classifier.")
    return task_kind


@register_atom(witness_partial_dependence_require_not_multiclass_multioutput)
@icontract.require(lambda is_classifier_task: _flag_valid(is_classifier_task), "is_classifier_task must be boolean")
@icontract.require(lambda classes_are_multioutput: _flag_valid(classes_are_multioutput), "classes_are_multioutput must be boolean")
@icontract.ensure(lambda result: _flag_valid(result), "result must be boolean")
def partial_dependence_require_not_multiclass_multioutput(
    *,
    is_classifier_task: bool,
    classes_are_multioutput: bool,
) -> bool:
    """Reject sklearn's unsupported multiclass-multioutput classifier configuration."""
    if is_classifier_task and classes_are_multioutput:
        raise ValueError("Multiclass-multioutput estimators are not supported")
    return classes_are_multioutput


@register_atom(witness_partial_dependence_resolve_recursion_response_method)
@icontract.require(lambda response_method: _response_method_valid(response_method), "response_method must be 'auto', 'predict_proba', or 'decision_function'")
@icontract.ensure(lambda result: _response_method_valid(result), "result must be a supported response method")
def partial_dependence_resolve_recursion_response_method(response_method: str) -> str:
    """Apply sklearn's recursion-specific response_method='auto' normalization."""
    if response_method == "auto":
        return "decision_function"
    return response_method


@register_atom(witness_partial_dependence_require_decision_function_for_recursion)
@icontract.require(lambda response_method: _response_method_valid(response_method), "response_method must be 'auto', 'predict_proba', or 'decision_function'")
@icontract.ensure(lambda result: _response_method_valid(result), "result must be a supported response method")
def partial_dependence_require_decision_function_for_recursion(response_method: str) -> str:
    """Require sklearn's decision_function-only response method for recursion mode."""
    if response_method != "decision_function":
        raise ValueError(
            "With the 'recursion' method, the response_method must be 'decision_function'. Got {}.".format(
                response_method
            )
        )
    return response_method
