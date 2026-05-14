"""Ghost witnesses for sklearn QuantileRegressor solver guard atoms."""

from __future__ import annotations


def witness_quantile_interior_point_removed_guard(solver: str, scipy_at_least_1_11: bool) -> bool:
    """Describe the SciPy-version guard for the interior-point solver."""
    return solver == "interior-point" and scipy_at_least_1_11


def witness_quantile_interior_point_removed_message(solver: str) -> str:
    """Describe the interior-point removal error message."""
    return f"Solver {solver} is not anymore available in SciPy >= 1.11.0."


def witness_quantile_sparse_solver_guard(is_sparse: bool, solver: str) -> bool:
    """Describe the sparse-X unsupported-solver guard."""
    return is_sparse and solver not in {"highs", "highs-ds", "highs-ipm"}


def witness_quantile_sparse_solver_message(solver: str) -> str:
    """Describe the sparse-X unsupported-solver error message."""
    return f"Solver {solver} does not support sparse X. Use solver 'highs' for example."


def witness_quantile_solver_options_payload(solver_options: object, solver: str) -> object:
    """Describe the solver_options payload passed toward linprog."""
    if solver_options is None and solver == "interior-point":
        return {"lstsq": True}
    return solver_options
