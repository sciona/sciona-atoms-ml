"""Sklearn coordinate-descent MultiTaskElasticNet estimator API-shell atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_multitask_elastic_net_constraints_without_unsupported,
    witness_cd_multitask_elastic_net_init_attributes,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


_UNSUPPORTED_PARAMS = frozenset({"precompute", "positive"})


@register_atom(witness_cd_multitask_elastic_net_constraints_without_unsupported)
@icontract.require(lambda parent_constraints: isinstance(parent_constraints, Mapping), "parent_constraints must be a mapping")
@icontract.require(lambda parent_constraints: _UNSUPPORTED_PARAMS.issubset(parent_constraints), "parent constraints must include precompute and positive")
@icontract.ensure(
    lambda result, parent_constraints: isinstance(result, dict)
    and _UNSUPPORTED_PARAMS.isdisjoint(result)
    and set(result) == (set(parent_constraints) - _UNSUPPORTED_PARAMS)
    and all(
        result[key] is value
        for key, value in parent_constraints.items()
        if key not in _UNSUPPORTED_PARAMS
    ),
    "MultiTaskElasticNet constraints must copy ElasticNet constraints and remove precompute and positive",
)
def cd_multitask_elastic_net_constraints_without_unsupported(
    parent_constraints: Mapping[str, object],
) -> dict[str, object]:
    """Return inherited ElasticNet constraints without unsupported parameters."""
    result = dict(parent_constraints)
    for param in _UNSUPPORTED_PARAMS:
        result.pop(param)
    return result


@register_atom(witness_cd_multitask_elastic_net_init_attributes)
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda warm_start: _bool(warm_start), "warm_start must be boolean")
@icontract.require(lambda selection: isinstance(selection, str), "selection must be a string")
@icontract.ensure(
    lambda result, alpha, l1_ratio, fit_intercept, copy_X, max_iter, tol, warm_start, random_state, selection: isinstance(result, dict)
    and set(result)
    == {
        "l1_ratio",
        "alpha",
        "fit_intercept",
        "max_iter",
        "copy_X",
        "tol",
        "warm_start",
        "random_state",
        "selection",
    }
    and result["l1_ratio"] is l1_ratio
    and result["alpha"] is alpha
    and result["fit_intercept"] is fit_intercept
    and result["max_iter"] is max_iter
    and result["copy_X"] is copy_X
    and result["tol"] is tol
    and result["warm_start"] is warm_start
    and result["random_state"] is random_state
    and result["selection"] is selection,
    "MultiTaskElasticNet init attributes must match sklearn assignment order and values",
)
def cd_multitask_elastic_net_init_attributes(
    alpha: object,
    l1_ratio: object,
    fit_intercept: bool,
    copy_X: bool,
    max_iter: int,
    tol: object,
    warm_start: bool,
    random_state: object,
    selection: str,
) -> dict[str, object]:
    """Return the attribute state assigned by MultiTaskElasticNet.__init__."""
    return {
        "l1_ratio": l1_ratio,
        "alpha": alpha,
        "fit_intercept": fit_intercept,
        "max_iter": max_iter,
        "copy_X": copy_X,
        "tol": tol,
        "warm_start": warm_start,
        "random_state": random_state,
        "selection": selection,
    }
