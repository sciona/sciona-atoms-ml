"""Ghost witnesses for sklearn multitask Lasso API atoms."""

from __future__ import annotations


def witness_cd_multitask_lasso_constraints_without_l1_ratio(
    parent_constraints: object,
) -> object:
    """Describe removing l1_ratio from inherited MultiTaskElasticNet constraints."""
    return parent_constraints


def witness_cd_multitask_lasso_fixed_l1_ratio(alpha: object) -> object:
    """Describe the fixed MultiTaskLasso l1_ratio value."""
    return alpha


def witness_cd_multitask_lasso_init_attributes(
    alpha: object,
    fit_intercept: object,
    copy_X: object,
    max_iter: object,
    tol: object,
    warm_start: object,
    random_state: object,
    selection: object,
) -> object:
    """Describe attributes assigned by MultiTaskLasso.__init__."""
    return (
        alpha,
        fit_intercept,
        copy_X,
        max_iter,
        tol,
        warm_start,
        random_state,
        selection,
    )
