"""Ghost witnesses for sklearn enet_path return-shell atoms."""

from __future__ import annotations


def witness_cd_enet_path_return_arity(return_n_iter: object) -> object:
    """Describe the final enet_path tuple arity selected by return_n_iter."""
    return return_n_iter


def witness_cd_enet_path_result_tuple(
    alphas: object,
    coefs: object,
    dual_gaps: object,
    n_iters: object,
    return_n_iter: object,
) -> object:
    """Describe the final public enet_path result tuple."""
    return alphas, coefs, dual_gaps, n_iters, return_n_iter
