"""Sklearn coordinate-descent multitask CV API-shell atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_multitask_elastic_net_cv_constraints,
    witness_cd_multitask_elastic_net_cv_init_attributes,
    witness_cd_multitask_lasso_cv_constraints_without_unsupported,
    witness_cd_multitask_lasso_cv_super_init_kwargs,
)


_UNSUPPORTED_PARAMS = frozenset({"precompute", "positive"})


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


@register_atom(witness_cd_multitask_elastic_net_cv_constraints)
@icontract.require(lambda parent_constraints: isinstance(parent_constraints, Mapping), "parent_constraints must be a mapping")
@icontract.require(lambda parent_constraints: _UNSUPPORTED_PARAMS.issubset(parent_constraints), "parent constraints must include precompute and positive")
@icontract.require(lambda l1_ratio_constraint: l1_ratio_constraint is not None, "l1_ratio_constraint must be supplied")
@icontract.ensure(
    lambda result, parent_constraints, l1_ratio_constraint: isinstance(result, dict)
    and _UNSUPPORTED_PARAMS.isdisjoint(result)
    and "l1_ratio" in result
    and result["l1_ratio"] is l1_ratio_constraint
    and set(result) == ((set(parent_constraints) - _UNSUPPORTED_PARAMS) | {"l1_ratio"})
    and all(
        result[key] is value
        for key, value in parent_constraints.items()
        if key not in _UNSUPPORTED_PARAMS and key != "l1_ratio"
    ),
    "MultiTaskElasticNetCV constraints must copy LinearModelCV constraints, add l1_ratio, and remove unsupported parameters",
)
def cd_multitask_elastic_net_cv_constraints(
    parent_constraints: Mapping[str, object],
    l1_ratio_constraint: object,
) -> dict[str, object]:
    """Return MultiTaskElasticNetCV parameter constraints."""
    result = dict(parent_constraints)
    result["l1_ratio"] = l1_ratio_constraint
    for param in _UNSUPPORTED_PARAMS:
        result.pop(param)
    return result


@register_atom(witness_cd_multitask_elastic_net_cv_init_attributes)
@icontract.require(lambda n_alphas: _positive_int(n_alphas), "n_alphas must be positive")
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.require(lambda selection: isinstance(selection, str), "selection must be a string")
@icontract.ensure(
    lambda result, l1_ratio, eps, n_alphas, alphas, fit_intercept, max_iter, tol, cv, copy_X, verbose, n_jobs, random_state, selection: isinstance(result, dict)
    and set(result)
    == {
        "l1_ratio",
        "eps",
        "n_alphas",
        "alphas",
        "fit_intercept",
        "max_iter",
        "tol",
        "cv",
        "copy_X",
        "verbose",
        "n_jobs",
        "random_state",
        "selection",
    }
    and result["l1_ratio"] is l1_ratio
    and result["eps"] is eps
    and result["n_alphas"] is n_alphas
    and result["alphas"] is alphas
    and result["fit_intercept"] is fit_intercept
    and result["max_iter"] is max_iter
    and result["tol"] is tol
    and result["cv"] is cv
    and result["copy_X"] is copy_X
    and result["verbose"] is verbose
    and result["n_jobs"] is n_jobs
    and result["random_state"] is random_state
    and result["selection"] is selection,
    "MultiTaskElasticNetCV init attributes must match sklearn assignment names and values",
)
def cd_multitask_elastic_net_cv_init_attributes(
    l1_ratio: object,
    eps: object,
    n_alphas: int,
    alphas: object,
    fit_intercept: bool,
    max_iter: int,
    tol: object,
    cv: object,
    copy_X: bool,
    verbose: object,
    n_jobs: object,
    random_state: object,
    selection: str,
) -> dict[str, object]:
    """Return the attribute state assigned by MultiTaskElasticNetCV.__init__."""
    return {
        "l1_ratio": l1_ratio,
        "eps": eps,
        "n_alphas": n_alphas,
        "alphas": alphas,
        "fit_intercept": fit_intercept,
        "max_iter": max_iter,
        "tol": tol,
        "cv": cv,
        "copy_X": copy_X,
        "verbose": verbose,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "selection": selection,
    }


@register_atom(witness_cd_multitask_lasso_cv_constraints_without_unsupported)
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
    "MultiTaskLassoCV constraints must copy LinearModelCV constraints and remove unsupported parameters",
)
def cd_multitask_lasso_cv_constraints_without_unsupported(
    parent_constraints: Mapping[str, object],
) -> dict[str, object]:
    """Return MultiTaskLassoCV parameter constraints without unsupported parameters."""
    result = dict(parent_constraints)
    for param in _UNSUPPORTED_PARAMS:
        result.pop(param)
    return result


@register_atom(witness_cd_multitask_lasso_cv_super_init_kwargs)
@icontract.require(lambda n_alphas: _positive_int(n_alphas), "n_alphas must be positive")
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.require(lambda selection: isinstance(selection, str), "selection must be a string")
@icontract.ensure(
    lambda result, eps, n_alphas, alphas, fit_intercept, max_iter, tol, copy_X, cv, verbose, n_jobs, random_state, selection: isinstance(result, dict)
    and result
    == {
        "eps": eps,
        "n_alphas": n_alphas,
        "alphas": alphas,
        "fit_intercept": fit_intercept,
        "max_iter": max_iter,
        "tol": tol,
        "copy_X": copy_X,
        "cv": cv,
        "verbose": verbose,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "selection": selection,
    },
    "MultiTaskLassoCV super-init kwargs must match sklearn delegation payload",
)
def cd_multitask_lasso_cv_super_init_kwargs(
    eps: object,
    n_alphas: int,
    alphas: object,
    fit_intercept: bool,
    max_iter: int,
    tol: object,
    copy_X: bool,
    cv: object,
    verbose: object,
    n_jobs: object,
    random_state: object,
    selection: str,
) -> dict[str, object]:
    """Return kwargs passed by MultiTaskLassoCV.__init__ into LinearModelCV.__init__."""
    return {
        "eps": eps,
        "n_alphas": n_alphas,
        "alphas": alphas,
        "fit_intercept": fit_intercept,
        "max_iter": max_iter,
        "tol": tol,
        "copy_X": copy_X,
        "cv": cv,
        "verbose": verbose,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "selection": selection,
    }
