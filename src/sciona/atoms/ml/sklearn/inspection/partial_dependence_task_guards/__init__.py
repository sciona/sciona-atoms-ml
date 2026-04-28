"""Partial-dependence task and recursion-response guard helper atoms."""

from .atoms import (
    partial_dependence_require_classifier_or_regressor,
    partial_dependence_require_decision_function_for_recursion,
    partial_dependence_require_not_multiclass_multioutput,
    partial_dependence_resolve_recursion_response_method,
)

__all__ = [
    "partial_dependence_require_classifier_or_regressor",
    "partial_dependence_require_decision_function_for_recursion",
    "partial_dependence_require_not_multiclass_multioutput",
    "partial_dependence_resolve_recursion_response_method",
]
