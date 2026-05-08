"""Sklearn coordinate-descent ElasticNetCV init atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_elastic_net_cv_constraints,
    witness_cd_elastic_net_cv_init_attributes,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


@register_atom(witness_cd_elastic_net_cv_constraints)
@icontract.require(lambda parent_constraints: isinstance(parent_constraints, Mapping), "parent_constraints must be a mapping")
@icontract.require(lambda l1_ratio_constraint: l1_ratio_constraint is not None, "l1_ratio_constraint must be supplied")
@icontract.ensure(
    lambda result, parent_constraints, l1_ratio_constraint: isinstance(result, dict)
    and "l1_ratio" in result
    and result["l1_ratio"] is l1_ratio_constraint
    and set(result) == (set(parent_constraints) | {"l1_ratio"})
    and all(
        result[key] is value
        for key, value in parent_constraints.items()
        if key != "l1_ratio"
    ),
    "ElasticNetCV constraints must copy LinearModelCV constraints and add l1_ratio",
)
def cd_elastic_net_cv_constraints(
    parent_constraints: Mapping[str, object],
    l1_ratio_constraint: object,
) -> dict[str, object]:
    """Return ElasticNetCV parameter constraints."""
    result = dict(parent_constraints)
    result["l1_ratio"] = l1_ratio_constraint
    return result


@register_atom(witness_cd_elastic_net_cv_init_attributes)
@icontract.require(lambda n_alphas: _positive_int(n_alphas), "n_alphas must be positive")
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.require(lambda positive: _bool(positive), "positive must be boolean")
@icontract.require(lambda selection: isinstance(selection, str), "selection must be a string")
@icontract.ensure(
    lambda result, l1_ratio, eps, n_alphas, alphas, fit_intercept, precompute, max_iter, tol, cv, copy_X, verbose, n_jobs, positive, random_state, selection: isinstance(result, dict)
    and set(result)
    == {
        "l1_ratio",
        "eps",
        "n_alphas",
        "alphas",
        "fit_intercept",
        "precompute",
        "max_iter",
        "tol",
        "cv",
        "copy_X",
        "verbose",
        "n_jobs",
        "positive",
        "random_state",
        "selection",
    }
    and result["l1_ratio"] is l1_ratio
    and result["eps"] is eps
    and result["n_alphas"] is n_alphas
    and result["alphas"] is alphas
    and result["fit_intercept"] is fit_intercept
    and result["precompute"] is precompute
    and result["max_iter"] is max_iter
    and result["tol"] is tol
    and result["cv"] is cv
    and result["copy_X"] is copy_X
    and result["verbose"] is verbose
    and result["n_jobs"] is n_jobs
    and result["positive"] is positive
    and result["random_state"] is random_state
    and result["selection"] is selection,
    "ElasticNetCV init attributes must match sklearn assignment names and values",
)
def cd_elastic_net_cv_init_attributes(
    l1_ratio: object,
    eps: object,
    n_alphas: int,
    alphas: object,
    fit_intercept: bool,
    precompute: object,
    max_iter: int,
    tol: object,
    cv: object,
    copy_X: bool,
    verbose: object,
    n_jobs: object,
    positive: bool,
    random_state: object,
    selection: str,
) -> dict[str, object]:
    """Return the attribute state assigned by ElasticNetCV.__init__."""
    return {
        "l1_ratio": l1_ratio,
        "eps": eps,
        "n_alphas": n_alphas,
        "alphas": alphas,
        "fit_intercept": fit_intercept,
        "precompute": precompute,
        "max_iter": max_iter,
        "tol": tol,
        "cv": cv,
        "copy_X": copy_X,
        "verbose": verbose,
        "n_jobs": n_jobs,
        "positive": positive,
        "random_state": random_state,
        "selection": selection,
    }
