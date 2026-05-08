"""Ghost witnesses for sklearn multitask solver result atoms."""

from __future__ import annotations


def witness_cd_multitask_solver_result_coef(solver_result: object) -> object:
    """Describe assigning solver result item 0 to `self.coef_`."""
    return solver_result


def witness_cd_multitask_solver_result_dual_gap(solver_result: object) -> object:
    """Describe assigning solver result item 1 to `self.dual_gap_`."""
    return solver_result


def witness_cd_multitask_solver_result_eps(solver_result: object) -> object:
    """Describe assigning solver result item 2 to `self.eps_`."""
    return solver_result


def witness_cd_multitask_solver_result_n_iter(solver_result: object) -> object:
    """Describe assigning solver result item 3 to `self.n_iter_`."""
    return solver_result


def witness_cd_multitask_set_intercept_args(
    X_offset: object,
    y_offset: object,
    X_scale: object,
) -> object:
    """Describe the `_set_intercept` positional payload in MultiTaskElasticNet.fit."""
    return X_offset, y_offset, X_scale
