"""Ghost witnesses for sklearn LinearModelCV init atoms."""

from __future__ import annotations


def witness_cd_cv_base_init_attributes(
    eps: object,
    n_alphas: object,
    alphas: object,
    fit_intercept: object,
    precompute: object,
    max_iter: object,
    tol: object,
    copy_X: object,
    cv: object,
    verbose: object,
    n_jobs: object,
    positive: object,
    random_state: object,
    selection: object,
) -> object:
    """Describe attributes assigned by LinearModelCV.__init__."""
    return (
        eps,
        n_alphas,
        alphas,
        fit_intercept,
        precompute,
        max_iter,
        tol,
        copy_X,
        cv,
        verbose,
        n_jobs,
        positive,
        random_state,
        selection,
    )
