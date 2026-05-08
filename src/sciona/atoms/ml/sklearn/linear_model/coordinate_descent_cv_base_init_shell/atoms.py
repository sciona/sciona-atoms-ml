"""Sklearn coordinate-descent LinearModelCV init atoms."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import witness_cd_cv_base_init_attributes


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


@register_atom(witness_cd_cv_base_init_attributes)
@icontract.require(lambda n_alphas: _positive_int(n_alphas), "n_alphas must be positive")
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.require(lambda positive: _bool(positive), "positive must be boolean")
@icontract.require(lambda selection: isinstance(selection, str), "selection must be a string")
@icontract.ensure(
    lambda result, eps, n_alphas, alphas, fit_intercept, precompute, max_iter, tol, copy_X, cv, verbose, n_jobs, positive, random_state, selection: isinstance(result, dict)
    and set(result)
    == {
        "eps",
        "n_alphas",
        "alphas",
        "fit_intercept",
        "precompute",
        "max_iter",
        "tol",
        "copy_X",
        "cv",
        "verbose",
        "n_jobs",
        "positive",
        "random_state",
        "selection",
    }
    and result["eps"] is eps
    and result["n_alphas"] is n_alphas
    and result["alphas"] is alphas
    and result["fit_intercept"] is fit_intercept
    and result["precompute"] is precompute
    and result["max_iter"] is max_iter
    and result["tol"] is tol
    and result["copy_X"] is copy_X
    and result["cv"] is cv
    and result["verbose"] is verbose
    and result["n_jobs"] is n_jobs
    and result["positive"] is positive
    and result["random_state"] is random_state
    and result["selection"] is selection,
    "LinearModelCV init attributes must match sklearn assignment names and values",
)
def cd_cv_base_init_attributes(
    eps: object,
    n_alphas: int,
    alphas: object,
    fit_intercept: bool,
    precompute: object,
    max_iter: int,
    tol: object,
    copy_X: bool,
    cv: object,
    verbose: object,
    n_jobs: object,
    positive: bool,
    random_state: object,
    selection: str,
) -> dict[str, object]:
    """Return the attribute state assigned by LinearModelCV.__init__."""
    return {
        "eps": eps,
        "n_alphas": n_alphas,
        "alphas": alphas,
        "fit_intercept": fit_intercept,
        "precompute": precompute,
        "max_iter": max_iter,
        "tol": tol,
        "copy_X": copy_X,
        "cv": cv,
        "verbose": verbose,
        "n_jobs": n_jobs,
        "positive": positive,
        "random_state": random_state,
        "selection": selection,
    }
