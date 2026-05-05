"""Partial-dependence brute response-method shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_brute_auto_response_method,
    witness_partial_dependence_brute_auto_target_method,
    witness_partial_dependence_brute_resolved_response_method,
)

VALID_RESPONSE_METHODS = {"auto", "predict_proba", "decision_function"}


def _response_method_valid(value: object) -> bool:
    return isinstance(value, str) and value in VALID_RESPONSE_METHODS


def _flag(value: object) -> bool:
    return isinstance(value, bool)


def _resolved_response_method_valid(value: object) -> bool:
    return value == "predict" or value == "predict_proba" or value == "decision_function" or value == (
        "predict_proba",
        "decision_function",
    )


@register_atom(witness_partial_dependence_brute_auto_response_method)
@icontract.require(lambda response_method: _response_method_valid(response_method), "response_method must be 'auto', 'predict_proba', or 'decision_function'")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_brute_auto_response_method(
    response_method: str,
) -> bool:
    """Decide whether _partial_dependence_brute normalizes response_method='auto'."""
    return response_method == "auto"


@register_atom(witness_partial_dependence_brute_auto_target_method)
@icontract.require(lambda is_regressor_task: _flag(is_regressor_task), "is_regressor_task must be boolean")
@icontract.ensure(lambda result: _resolved_response_method_valid(result), "result must be a valid brute response-method target")
def partial_dependence_brute_auto_target_method(
    *,
    is_regressor_task: bool,
) -> str | tuple[str, str]:
    """Resolve sklearn's auto response-method target for brute partial dependence."""
    if is_regressor_task:
        return "predict"
    return ("predict_proba", "decision_function")


@register_atom(witness_partial_dependence_brute_resolved_response_method)
@icontract.require(lambda response_method: _response_method_valid(response_method), "response_method must be 'auto', 'predict_proba', or 'decision_function'")
@icontract.require(lambda is_regressor_task: _flag(is_regressor_task), "is_regressor_task must be boolean")
@icontract.ensure(lambda result: _resolved_response_method_valid(result), "result must be a valid brute response-method target")
def partial_dependence_brute_resolved_response_method(
    response_method: str,
    *,
    is_regressor_task: bool,
) -> str | tuple[str, str]:
    """Resolve sklearn's brute response_method before estimator callbacks."""
    if response_method == "auto":
        return partial_dependence_brute_auto_target_method(
            is_regressor_task=is_regressor_task
        )
    return response_method
