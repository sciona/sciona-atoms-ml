"""Ghost witnesses for partial-dependence task and recursion-response guard atoms."""

from __future__ import annotations


def witness_partial_dependence_require_classifier_or_regressor(task_kind: str) -> str:
    """Describe the supported task-kind guard for partial dependence."""
    return task_kind


def witness_partial_dependence_require_not_multiclass_multioutput(
    *,
    is_classifier_task: bool,
    classes_are_multioutput: bool,
) -> bool:
    """Describe the multiclass-multioutput rejection guard."""
    del is_classifier_task
    return classes_are_multioutput


def witness_partial_dependence_resolve_recursion_response_method(response_method: str) -> str:
    """Describe recursion-mode response_method normalization."""
    return response_method


def witness_partial_dependence_require_decision_function_for_recursion(response_method: str) -> str:
    """Describe the recursion-mode decision_function guard."""
    return response_method
