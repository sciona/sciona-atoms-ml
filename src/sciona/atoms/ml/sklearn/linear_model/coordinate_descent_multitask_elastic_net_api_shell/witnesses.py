"""Ghost witnesses for sklearn multi-output ElasticNet API atoms."""

from __future__ import annotations


def witness_cd_multitask_elastic_net_constraints_without_unsupported(
    parent_constraints: object,
) -> object:
    """Describe removing unsupported parameters from inherited constraints."""
    return parent_constraints


def witness_cd_multitask_elastic_net_init_attributes(
    alpha: object,
    l1_ratio: object,
    fit_intercept: object,
    copy_X: object,
    max_iter: object,
    tol: object,
    warm_start: object,
    random_state: object,
    selection: object,
) -> object:
    """Describe attributes assigned by MultiTaskElasticNet.__init__."""
    return (
        alpha,
        l1_ratio,
        fit_intercept,
        copy_X,
        max_iter,
        tol,
        warm_start,
        random_state,
        selection,
    )
