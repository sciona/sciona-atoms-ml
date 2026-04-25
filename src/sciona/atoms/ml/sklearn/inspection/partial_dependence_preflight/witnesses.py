"""Ghost witnesses for partial-dependence preflight helper atoms."""

from __future__ import annotations


def witness_partial_dependence_require_response_method_auto_for_regressor(
    task_kind: str,
    response_method: str,
) -> str:
    """Describe the response-method guard before partial-dependence execution."""
    del task_kind
    return response_method


def witness_partial_dependence_resolve_kind_method(
    kind: str,
    method: str,
) -> str:
    """Describe the effective method after sklearn's kind-versus-method rule."""
    del kind
    return method


def witness_partial_dependence_require_no_sample_weight_for_recursion(
    method: str,
    *,
    sample_weight_provided: bool,
) -> str:
    """Describe the sample-weight restriction for recursion mode."""
    del sample_weight_provided
    return method


def witness_partial_dependence_resolve_auto_method(
    method: str,
    *,
    sample_weight_provided: bool,
    supports_recursion: bool,
) -> str:
    """Describe the effective method after sklearn's method='auto' resolution."""
    del sample_weight_provided
    del supports_recursion
    return method


def witness_partial_dependence_require_recursion_support(
    method: str,
    *,
    supports_recursion: bool,
) -> str:
    """Describe the recursion-support guard for supported estimator families."""
    del supports_recursion
    return method
