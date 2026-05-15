"""Sklearn LARS CV orchestration shell atoms."""

from __future__ import annotations

from numbers import Integral, Real

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import witness_lars_cv_path_residues_callback_kwargs

_PATH_RESIDUES_CALLBACK_KEYS = {
    "Gram",
    "copy",
    "method",
    "verbose",
    "fit_intercept",
    "max_iter",
    "eps",
    "positive",
}


def _valid_lars_cv_method(method: object) -> bool:
    return bool(method in {"lar", "lasso"})


def _valid_verbose(value: object) -> bool:
    return bool(
        isinstance(value, (Integral, bool, np.integer, np.bool_))
        and int(value) >= 0
    )


def _positive_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0)


def _finite_positive_real(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)


def _bool_like(value: object) -> bool:
    return bool(isinstance(value, (bool, np.bool_)))


def _path_residues_kwargs_valid(
    result: dict[str, object],
    precompute: object,
    method: str,
    verbose: int | bool,
    fit_intercept: bool,
    max_iter: int,
    eps: float,
    positive: bool,
) -> bool:
    return bool(
        set(result) == _PATH_RESIDUES_CALLBACK_KEYS
        and result["Gram"] is precompute
        and result["copy"] is False
        and result["method"] == method
        and result["verbose"] == max(0, int(verbose) - 1)
        and result["fit_intercept"] is fit_intercept
        and result["max_iter"] == int(max_iter)
        and result["eps"] == float(eps)
        and result["positive"] is positive
    )


@register_atom(witness_lars_cv_path_residues_callback_kwargs)
@icontract.require(lambda method: _valid_lars_cv_method(method), "method must be the sklearn LARS CV path method")
@icontract.require(lambda verbose: _valid_verbose(verbose), "verbose must be a nonnegative bool/int verbosity context")
@icontract.require(lambda fit_intercept: _bool_like(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda max_iter: _positive_integer(max_iter), "max_iter must be a positive integer")
@icontract.require(lambda eps: _finite_positive_real(eps), "eps must be finite and positive")
@icontract.require(lambda positive: _bool_like(positive), "positive must be boolean")
@icontract.ensure(
    lambda result, precompute, method, verbose, fit_intercept, max_iter, eps, positive: _path_residues_kwargs_valid(
        result,
        precompute,
        method,
        verbose,
        fit_intercept,
        max_iter,
        eps,
        positive,
    ),
    "_lars_path_residues kwargs must match LarsCV.fit callback payload",
)
def lars_cv_path_residues_callback_kwargs(
    *,
    precompute: object,
    method: str,
    verbose: int | bool = 0,
    fit_intercept: bool = True,
    max_iter: int = 500,
    eps: float = np.finfo(float).eps,
    positive: bool = False,
) -> dict[str, object]:
    """Return the kwargs LarsCV.fit passes to each _lars_path_residues fold callback."""
    return {
        "Gram": precompute,
        "copy": False,
        "method": method,
        "verbose": max(0, int(verbose) - 1),
        "fit_intercept": fit_intercept,
        "max_iter": int(max_iter),
        "eps": float(eps),
        "positive": positive,
    }
