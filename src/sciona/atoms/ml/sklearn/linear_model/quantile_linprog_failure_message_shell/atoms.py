"""Sklearn QuantileRegressor linprog failure-message atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_quantile_linprog_failure_guard,
    witness_quantile_linprog_failure_message,
    witness_quantile_linprog_failure_reason,
)

_FAILURE_REASONS = {
    1: "Iteration limit reached.",
    2: "Problem appears to be infeasible.",
    3: "Problem appears to be unbounded.",
    4: "Numerical difficulties encountered.",
}


def _status_valid(status: int) -> bool:
    return isinstance(status, int) and not isinstance(status, bool)


def _message_valid(message: str) -> bool:
    return isinstance(message, str)


@register_atom(witness_quantile_linprog_failure_guard)
@icontract.require(lambda success: isinstance(success, bool), "success must be boolean")
@icontract.ensure(lambda result, success: result is (not success), "failure branch applies only when result.success is false")
def quantile_linprog_failure_guard(success: bool) -> bool:
    """Return whether QuantileRegressor should enter the linprog failure branch."""
    return not success


@register_atom(witness_quantile_linprog_failure_reason)
@icontract.require(lambda status: _status_valid(status), "status must be an integer")
@icontract.ensure(
    lambda result, status: result == _FAILURE_REASONS.get(status, "unknown reason"),
    "failure reason must match sklearn status mapping",
)
def quantile_linprog_failure_reason(status: int) -> str:
    """Return the QuantileRegressor failure reason for a linprog status."""
    return _FAILURE_REASONS.get(status, "unknown reason")


@register_atom(witness_quantile_linprog_failure_message)
@icontract.require(lambda status: _status_valid(status), "status must be an integer")
@icontract.require(lambda result_message: _message_valid(result_message), "result_message must be a string")
@icontract.ensure(
    lambda result, status, result_message: result
    == (
        "Linear programming for QuantileRegressor did not succeed.\n"
        f"Status is {status}: "
        + quantile_linprog_failure_reason(status)
        + "\n"
        + "Result message of linprog:\n"
        + result_message
    ),
    "failure message must match sklearn warning payload",
)
def quantile_linprog_failure_message(status: int, result_message: str) -> str:
    """Return the QuantileRegressor unsuccessful-linprog warning message."""
    return (
        "Linear programming for QuantileRegressor did not succeed.\n"
        f"Status is {status}: "
        + quantile_linprog_failure_reason(status)
        + "\n"
        + "Result message of linprog:\n"
        + result_message
    )
