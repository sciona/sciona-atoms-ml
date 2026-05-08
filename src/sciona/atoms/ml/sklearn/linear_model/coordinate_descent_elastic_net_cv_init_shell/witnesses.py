"""Ghost witnesses for sklearn ElasticNetCV init atoms."""

from __future__ import annotations


def witness_cd_elastic_net_cv_constraints(
    parent_constraints: object,
    l1_ratio_constraint: object,
) -> object:
    """Describe ElasticNetCV parameter-constraint specialization."""
    return parent_constraints, l1_ratio_constraint


def witness_cd_elastic_net_cv_init_attributes(
    l1_ratio: object,
    eps: object,
    n_alphas: object,
    alphas: object,
    fit_intercept: object,
    precompute: object,
    max_iter: object,
    tol: object,
    cv: object,
    copy_X: object,
    verbose: object,
    n_jobs: object,
    positive: object,
    random_state: object,
    selection: object,
) -> object:
    """Describe ElasticNetCV direct __init__ assignments."""
    return (
        l1_ratio,
        eps,
        n_alphas,
        alphas,
        fit_intercept,
        precompute,
        max_iter,
        tol,
        cv,
        copy_X,
        verbose,
        n_jobs,
        positive,
        random_state,
        selection,
    )
