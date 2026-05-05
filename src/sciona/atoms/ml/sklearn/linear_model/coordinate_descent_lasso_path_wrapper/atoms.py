"""Sklearn coordinate-descent lasso_path wrapper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_lasso_path_call_kwargs,
    witness_cd_lasso_path_result,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and value >= 1


@register_atom(witness_cd_lasso_path_call_kwargs)
@icontract.require(lambda eps: float(eps) > 0.0, "eps must be positive")
@icontract.require(lambda n_alphas: _positive_int(n_alphas), "n_alphas must be positive")
@icontract.require(lambda copy_X: isinstance(copy_X, bool), "copy_X must be boolean")
@icontract.require(lambda positive: isinstance(positive, bool), "positive must be boolean")
@icontract.require(lambda return_n_iter: isinstance(return_n_iter, bool), "return_n_iter must be boolean")
@icontract.require(lambda params: isinstance(params, Mapping), "params must be a mapping")
@icontract.ensure(
    lambda result, eps, n_alphas, alphas, precompute, Xy, copy_X, coef_init, verbose, positive, return_n_iter, params:
    result == {
        "l1_ratio": 1.0,
        "eps": eps,
        "n_alphas": n_alphas,
        "alphas": alphas,
        "precompute": precompute,
        "Xy": Xy,
        "copy_X": copy_X,
        "coef_init": coef_init,
        "verbose": verbose,
        "positive": positive,
        "return_n_iter": return_n_iter,
        **dict(params),
    },
    "call kwargs must match the fixed lasso_path enet_path delegation shell",
)
def cd_lasso_path_call_kwargs(
    eps: float,
    n_alphas: int,
    alphas: object,
    precompute: object,
    Xy: object,
    copy_X: bool,
    coef_init: object,
    verbose: object,
    positive: bool,
    return_n_iter: bool,
    params: Mapping[str, object],
) -> dict[str, object]:
    """Return the enet_path kwargs assembled by lasso_path."""
    return {
        "l1_ratio": 1.0,
        "eps": eps,
        "n_alphas": n_alphas,
        "alphas": alphas,
        "precompute": precompute,
        "Xy": Xy,
        "copy_X": copy_X,
        "coef_init": coef_init,
        "verbose": verbose,
        "positive": positive,
        "return_n_iter": return_n_iter,
        **dict(params),
    }


@register_atom(witness_cd_lasso_path_result)
@icontract.require(lambda return_n_iter: isinstance(return_n_iter, bool), "return_n_iter must be boolean")
@icontract.require(
    lambda delegated_result: isinstance(delegated_result, tuple) and len(delegated_result) in {3, 4},
    "delegated_result must be a three- or four-item enet_path tuple",
)
@icontract.ensure(
    lambda result, delegated_result, return_n_iter: result == delegated_result
    and len(result) == (4 if return_n_iter else 3),
    "lasso_path must return the delegated enet_path tuple unchanged",
)
def cd_lasso_path_result(delegated_result: tuple[object, ...], return_n_iter: bool) -> tuple[object, ...]:
    """Return the final result tuple passed through by lasso_path."""
    return delegated_result
