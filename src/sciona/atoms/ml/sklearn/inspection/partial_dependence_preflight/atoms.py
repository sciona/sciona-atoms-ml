"""Partial-dependence preflight helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_require_no_sample_weight_for_recursion,
    witness_partial_dependence_require_recursion_support,
    witness_partial_dependence_require_response_method_auto_for_regressor,
    witness_partial_dependence_resolve_auto_method,
    witness_partial_dependence_resolve_kind_method,
)

VALID_TASK_KINDS = {"classifier", "regressor"}
VALID_RESPONSE_METHODS = {"auto", "predict_proba", "decision_function"}
VALID_METHODS = {"auto", "brute", "recursion"}
VALID_KINDS = {"average", "individual", "both"}


def _task_kind_valid(task_kind: object) -> bool:
    return isinstance(task_kind, str) and task_kind in VALID_TASK_KINDS


def _response_method_valid(response_method: object) -> bool:
    return isinstance(response_method, str) and response_method in VALID_RESPONSE_METHODS


def _method_valid(method: object) -> bool:
    return isinstance(method, str) and method in VALID_METHODS


def _kind_valid(kind: object) -> bool:
    return isinstance(kind, str) and kind in VALID_KINDS


def _flag_valid(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_partial_dependence_require_response_method_auto_for_regressor)
@icontract.require(lambda task_kind: _task_kind_valid(task_kind), "task_kind must be 'classifier' or 'regressor'")
@icontract.require(lambda response_method: _response_method_valid(response_method), "response_method must be a supported partial-dependence response method")
@icontract.ensure(lambda result: _response_method_valid(result), "result must be a supported response method")
def partial_dependence_require_response_method_auto_for_regressor(
    task_kind: str,
    response_method: str,
) -> str:
    """Enforce sklearn's regressor-side response_method restriction for partial dependence."""
    if task_kind == "regressor" and response_method != "auto":
        raise ValueError(
            "The response_method parameter is ignored for regressors and must be 'auto'."
        )
    return response_method


@register_atom(witness_partial_dependence_resolve_kind_method)
@icontract.require(lambda kind: _kind_valid(kind), "kind must be 'average', 'individual', or 'both'")
@icontract.require(lambda method: _method_valid(method), "method must be 'auto', 'brute', or 'recursion'")
@icontract.ensure(lambda result: _method_valid(result), "result must be a supported partial-dependence method")
def partial_dependence_resolve_kind_method(
    kind: str,
    method: str,
) -> str:
    """Apply sklearn's kind-versus-method rule before auto-method resolution."""
    if kind != "average":
        if method == "recursion":
            raise ValueError(
                "The 'recursion' method only applies when 'kind' is set to 'average'"
            )
        return "brute"
    return method


@register_atom(witness_partial_dependence_require_no_sample_weight_for_recursion)
@icontract.require(lambda method: _method_valid(method), "method must be 'auto', 'brute', or 'recursion'")
@icontract.require(lambda sample_weight_provided: _flag_valid(sample_weight_provided), "sample_weight_provided must be boolean")
@icontract.ensure(lambda result: _method_valid(result), "result must be a supported partial-dependence method")
def partial_dependence_require_no_sample_weight_for_recursion(
    method: str,
    *,
    sample_weight_provided: bool,
) -> str:
    """Enforce sklearn's recursion restriction when sample weights are present."""
    if method == "recursion" and sample_weight_provided:
        raise ValueError(
            "The 'recursion' method can only be applied when sample_weight is None."
        )
    return method


@register_atom(witness_partial_dependence_resolve_auto_method)
@icontract.require(lambda method: _method_valid(method), "method must be 'auto', 'brute', or 'recursion'")
@icontract.require(lambda sample_weight_provided: _flag_valid(sample_weight_provided), "sample_weight_provided must be boolean")
@icontract.require(lambda supports_recursion: _flag_valid(supports_recursion), "supports_recursion must be boolean")
@icontract.ensure(lambda result: _method_valid(result), "result must be a supported partial-dependence method")
def partial_dependence_resolve_auto_method(
    method: str,
    *,
    sample_weight_provided: bool,
    supports_recursion: bool,
) -> str:
    """Resolve sklearn's method='auto' branch from supplied recursion-support and sample-weight flags."""
    if method != "auto":
        return method
    if sample_weight_provided:
        return "brute"
    if supports_recursion:
        return "recursion"
    return "brute"


@register_atom(witness_partial_dependence_require_recursion_support)
@icontract.require(lambda method: _method_valid(method), "method must be 'auto', 'brute', or 'recursion'")
@icontract.require(lambda supports_recursion: _flag_valid(supports_recursion), "supports_recursion must be boolean")
@icontract.ensure(lambda result: _method_valid(result), "result must be a supported partial-dependence method")
def partial_dependence_require_recursion_support(
    method: str,
    *,
    supports_recursion: bool,
) -> str:
    """Reject recursion when the estimator family is not marked recursion-supported."""
    if method == "recursion" and not supports_recursion:
        raise ValueError(
            "Only recursion-supported estimators can use the 'recursion' method. Try using method='brute'."
        )
    return method
