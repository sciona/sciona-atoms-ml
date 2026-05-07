"""Sklearn coordinate-descent Lasso estimator API-shell atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_lasso_constraints_without_l1_ratio,
    witness_cd_lasso_fixed_l1_ratio,
    witness_cd_lasso_path_name,
    witness_cd_lasso_super_init_kwargs,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


@register_atom(witness_cd_lasso_constraints_without_l1_ratio)
@icontract.require(lambda parent_constraints: isinstance(parent_constraints, Mapping), "parent_constraints must be a mapping")
@icontract.require(lambda parent_constraints: "l1_ratio" in parent_constraints, "parent constraints must include l1_ratio")
@icontract.ensure(
    lambda result, parent_constraints: isinstance(result, dict)
    and "l1_ratio" not in result
    and set(result) == (set(parent_constraints) - {"l1_ratio"})
    and all(
        result[key] is value
        for key, value in parent_constraints.items()
        if key != "l1_ratio"
    ),
    "Lasso constraints must copy ElasticNet constraints and remove l1_ratio",
)
def cd_lasso_constraints_without_l1_ratio(
    parent_constraints: Mapping[str, object],
) -> dict[str, object]:
    """Return inherited ElasticNet parameter constraints without l1_ratio."""
    result = dict(parent_constraints)
    result.pop("l1_ratio")
    return result


@register_atom(witness_cd_lasso_path_name)
@icontract.require(lambda estimator_kind: estimator_kind == "lasso", "estimator_kind must be lasso")
@icontract.ensure(lambda result: result == "enet_path", "Lasso.path must be the enet_path helper")
def cd_lasso_path_name(estimator_kind: str) -> str:
    """Return the path helper name selected by Lasso."""
    del estimator_kind
    return "enet_path"


@register_atom(witness_cd_lasso_fixed_l1_ratio)
@icontract.ensure(
    lambda result, alpha: isinstance(result, float) and result == 1.0,
    "Lasso forwards fixed l1_ratio=1.0",
)
def cd_lasso_fixed_l1_ratio(alpha: object) -> float:
    """Return the fixed l1_ratio forwarded by Lasso.__init__."""
    del alpha
    return 1.0


@register_atom(witness_cd_lasso_super_init_kwargs)
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda warm_start: _bool(warm_start), "warm_start must be boolean")
@icontract.require(lambda positive: _bool(positive), "positive must be boolean")
@icontract.require(lambda selection: isinstance(selection, str), "selection must be a string")
@icontract.ensure(
    lambda result, alpha, fit_intercept, precompute, copy_X, max_iter, tol, warm_start, positive, random_state, selection: isinstance(result, dict)
    and set(result)
    == {
        "alpha",
        "l1_ratio",
        "fit_intercept",
        "precompute",
        "copy_X",
        "max_iter",
        "tol",
        "warm_start",
        "positive",
        "random_state",
        "selection",
    }
    and result["alpha"] is alpha
    and result["l1_ratio"] == 1.0
    and result["fit_intercept"] is fit_intercept
    and result["precompute"] is precompute
    and result["copy_X"] is copy_X
    and result["max_iter"] is max_iter
    and result["tol"] is tol
    and result["warm_start"] is warm_start
    and result["positive"] is positive
    and result["random_state"] is random_state
    and result["selection"] is selection,
    "Lasso.__init__ kwargs must match the ElasticNet.__init__ delegation",
)
def cd_lasso_super_init_kwargs(
    alpha: object,
    fit_intercept: bool,
    precompute: object,
    copy_X: bool,
    max_iter: int,
    tol: object,
    warm_start: bool,
    positive: bool,
    random_state: object,
    selection: str,
) -> dict[str, object]:
    """Return kwargs passed from Lasso.__init__ into ElasticNet.__init__."""
    return {
        "alpha": alpha,
        "l1_ratio": 1.0,
        "fit_intercept": fit_intercept,
        "precompute": precompute,
        "copy_X": copy_X,
        "max_iter": max_iter,
        "tol": tol,
        "warm_start": warm_start,
        "positive": positive,
        "random_state": random_state,
        "selection": selection,
    }
