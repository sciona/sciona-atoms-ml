"""Ghost witnesses for sklearn multitask CV API-shell atoms."""

from __future__ import annotations


def witness_cd_multitask_elastic_net_cv_constraints(
    parent_constraints: object, l1_ratio_constraint: object
) -> object:
    """Describe MultiTaskElasticNetCV parameter-constraint specialization."""
    return parent_constraints, l1_ratio_constraint


def witness_cd_multitask_elastic_net_cv_init_attributes(
    l1_ratio: object,
    eps: object,
    n_alphas: object,
    alphas: object,
    fit_intercept: object,
    max_iter: object,
    tol: object,
    cv: object,
    copy_X: object,
    verbose: object,
    n_jobs: object,
    random_state: object,
    selection: object,
) -> object:
    """Describe MultiTaskElasticNetCV direct __init__ assignments."""
    return (
        l1_ratio,
        eps,
        n_alphas,
        alphas,
        fit_intercept,
        max_iter,
        tol,
        cv,
        copy_X,
        verbose,
        n_jobs,
        random_state,
        selection,
    )


def witness_cd_multitask_lasso_cv_constraints_without_unsupported(
    parent_constraints: object,
) -> object:
    """Describe MultiTaskLassoCV parameter-constraint pruning."""
    return parent_constraints


def witness_cd_multitask_lasso_cv_super_init_kwargs(
    eps: object,
    n_alphas: object,
    alphas: object,
    fit_intercept: object,
    max_iter: object,
    tol: object,
    copy_X: object,
    cv: object,
    verbose: object,
    n_jobs: object,
    random_state: object,
    selection: object,
) -> object:
    """Describe MultiTaskLassoCV delegation kwargs into LinearModelCV.__init__."""
    return (
        eps,
        n_alphas,
        alphas,
        fit_intercept,
        max_iter,
        tol,
        copy_X,
        cv,
        verbose,
        n_jobs,
        random_state,
        selection,
    )
