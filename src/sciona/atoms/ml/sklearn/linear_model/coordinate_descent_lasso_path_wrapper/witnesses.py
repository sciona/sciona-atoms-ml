"""Ghost witnesses for sklearn coordinate-descent lasso_path wrapper atoms."""

from __future__ import annotations


def witness_cd_lasso_path_call_kwargs(
    eps: object,
    n_alphas: object,
    alphas: object,
    precompute: object,
    Xy: object,
    copy_X: object,
    coef_init: object,
    verbose: object,
    positive: object,
    return_n_iter: object,
    params: object,
) -> object:
    """Describe the fixed enet_path keyword shell in lasso_path."""
    return (
        eps,
        n_alphas,
        alphas,
        precompute,
        Xy,
        copy_X,
        coef_init,
        verbose,
        positive,
        return_n_iter,
        params,
    )


def witness_cd_lasso_path_result(delegated_result: object, return_n_iter: object) -> object:
    """Describe the final return passthrough shell in lasso_path."""
    return delegated_result, return_n_iter
