"""Ghost witnesses for sklearn QuantileRegressor linprog callback atoms."""

from __future__ import annotations


def witness_quantile_linprog_callback_payload(c: object, A_eq: object, b_eq: object, solver: str, solver_options: object) -> dict[str, object]:
    """Describe the keyword payload passed to scipy.optimize.linprog."""
    return {"c": c, "A_eq": A_eq, "b_eq": b_eq, "method": solver, "options": solver_options}


def witness_quantile_linprog_solution(result: object) -> object:
    """Describe extraction of the raw solution vector from the linprog result."""
    return result.x
