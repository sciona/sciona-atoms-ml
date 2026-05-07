"""Ghost witnesses for sklearn coordinate-descent Lasso estimator API atoms."""

from __future__ import annotations


def witness_cd_lasso_constraints_without_l1_ratio(parent_constraints: object) -> object:
    """Describe removing l1_ratio from inherited ElasticNet constraints."""
    return parent_constraints


def witness_cd_lasso_path_name(estimator_kind: object) -> object:
    """Describe Lasso path helper selection."""
    return estimator_kind


def witness_cd_lasso_fixed_l1_ratio(alpha: object) -> object:
    """Describe the fixed Lasso l1_ratio value."""
    return alpha


def witness_cd_lasso_super_init_kwargs(
    alpha: object,
    fit_intercept: object,
    precompute: object,
    copy_X: object,
    max_iter: object,
    tol: object,
    warm_start: object,
    positive: object,
    random_state: object,
    selection: object,
) -> object:
    """Describe kwargs forwarded from Lasso.__init__ to ElasticNet.__init__."""
    return (
        alpha,
        fit_intercept,
        precompute,
        copy_X,
        max_iter,
        tol,
        warm_start,
        positive,
        random_state,
        selection,
    )
