"""Ghost witnesses for sklearn QuantileRegressor linprog failure messages."""

from __future__ import annotations


def witness_quantile_linprog_failure_guard(success: bool) -> bool:
    """Describe whether the unsuccessful-linprog warning branch applies."""
    return not success


def witness_quantile_linprog_failure_reason(status: int) -> str:
    """Describe the fixed failure reason selected from linprog status."""
    failure = {
        1: "Iteration limit reached.",
        2: "Problem appears to be infeasible.",
        3: "Problem appears to be unbounded.",
        4: "Numerical difficulties encountered.",
    }
    return failure.get(status, "unknown reason")


def witness_quantile_linprog_failure_message(status: int, result_message: str) -> str:
    """Describe the full QuantileRegressor linprog warning message."""
    reason = witness_quantile_linprog_failure_reason(status)
    return (
        "Linear programming for QuantileRegressor did not succeed.\n"
        f"Status is {status}: "
        + reason
        + "\n"
        + "Result message of linprog:\n"
        + result_message
    )
