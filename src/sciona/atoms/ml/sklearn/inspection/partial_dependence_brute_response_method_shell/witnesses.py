"""Ghost witnesses for partial-dependence brute response-method shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_brute_auto_response_method(
    response_method: str,
) -> AbstractArray:
    """Describe the response_method='auto' predicate in _partial_dependence_brute."""
    del response_method
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_brute_auto_target_method(
    *,
    is_regressor_task: bool,
) -> AbstractArray | tuple[str, str]:
    """Describe sklearn's auto response-method target."""
    del is_regressor_task
    return AbstractArray(shape=(), dtype="object")


def witness_partial_dependence_brute_resolved_response_method(
    response_method: str,
    *,
    is_regressor_task: bool,
) -> AbstractArray | tuple[str, str]:
    """Describe the resolved brute response method before estimator callbacks."""
    del response_method, is_regressor_task
    return AbstractArray(shape=(), dtype="object")
