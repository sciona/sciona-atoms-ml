"""Sklearn LARS CV final refit callback atoms."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_lars_cv_fit_return_self,
    witness_lars_cv_refit_fit_call,
    witness_lars_cv_refit_fit_kwargs,
    witness_lars_cv_refit_state_payload,
)

_STATE_KEYS = {"alpha_", "cv_alphas_", "mse_path_"}
_FIT_KWARGS_KEYS = {"max_iter", "alpha", "Xy", "fit_path"}
_FIT_CALL_KEYS = {"args", "kwargs"}


def _finite_real(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(float(value)))


def _nonnegative_real(value: object) -> bool:
    return bool(_finite_real(value) and float(value) >= 0.0)


def _positive_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0)


def _not_none(value: object) -> bool:
    return value is not None


def _state_payload_valid(result: Mapping[str, object], best_alpha: float, cv_alphas: object, mse_path: object) -> bool:
    return bool(
        set(result) == _STATE_KEYS
        and result["alpha_"] == float(best_alpha)
        and result["cv_alphas_"] is cv_alphas
        and result["mse_path_"] is mse_path
    )


def _fit_kwargs_valid(result: Mapping[str, object], max_iter: int, best_alpha: float) -> bool:
    return bool(
        set(result) == _FIT_KWARGS_KEYS
        and result["max_iter"] == int(max_iter)
        and result["alpha"] == float(best_alpha)
        and result["Xy"] is None
        and result["fit_path"] is True
    )


def _refit_kwargs_mapping_valid(value: object) -> bool:
    return bool(isinstance(value, Mapping) and set(value) == _FIT_KWARGS_KEYS and value.get("Xy") is None and value.get("fit_path") is True)


def _fit_call_valid(result: Mapping[str, object], X: object, y: object, kwargs: object) -> bool:
    args = result.get("args")
    return bool(
        set(result) == _FIT_CALL_KEYS
        and isinstance(args, tuple)
        and len(args) == 2
        and args[0] is X
        and args[1] is y
        and result["kwargs"] is kwargs
    )


@register_atom(witness_lars_cv_refit_state_payload)
@icontract.require(lambda best_alpha: _nonnegative_real(best_alpha), "best_alpha must be finite and nonnegative")
@icontract.require(lambda cv_alphas: _not_none(cv_alphas), "cv_alphas must be supplied")
@icontract.require(lambda mse_path: _not_none(mse_path), "mse_path must be supplied")
@icontract.ensure(
    lambda result, best_alpha, cv_alphas, mse_path: _state_payload_valid(result, best_alpha, cv_alphas, mse_path),
    "selected CV state payload must match LarsCV.fit assignments",
)
def lars_cv_refit_state_payload(best_alpha: float, cv_alphas: object, mse_path: object) -> dict[str, object]:
    """Return the selected state payload stored before LarsCV's final refit."""
    return {"alpha_": float(best_alpha), "cv_alphas_": cv_alphas, "mse_path_": mse_path}


@register_atom(witness_lars_cv_refit_fit_kwargs)
@icontract.require(lambda max_iter: _positive_integer(max_iter), "max_iter must be a positive integer")
@icontract.require(lambda best_alpha: _nonnegative_real(best_alpha), "best_alpha must be finite and nonnegative")
@icontract.ensure(
    lambda result, max_iter, best_alpha: _fit_kwargs_valid(result, max_iter, best_alpha),
    "final _fit kwargs must match LarsCV.fit refit callback",
)
def lars_cv_refit_fit_kwargs(max_iter: int, best_alpha: float) -> dict[str, object]:
    """Return kwargs passed to LarsCV._fit for the final refit."""
    return {"max_iter": int(max_iter), "alpha": float(best_alpha), "Xy": None, "fit_path": True}


@register_atom(witness_lars_cv_refit_fit_call)
@icontract.require(lambda X: _not_none(X), "X must be supplied")
@icontract.require(lambda y: _not_none(y), "y must be supplied")
@icontract.require(lambda kwargs: _refit_kwargs_mapping_valid(kwargs), "kwargs must match the LarsCV final refit payload")
@icontract.ensure(
    lambda result, X, y, kwargs: _fit_call_valid(result, X, y, kwargs),
    "final _fit call payload must preserve positional and keyword identities",
)
def lars_cv_refit_fit_call(X: object, y: object, kwargs: Mapping[str, object]) -> dict[str, object]:
    """Return the positional and keyword payload for LarsCV._fit."""
    return {"args": (X, y), "kwargs": kwargs}


@register_atom(witness_lars_cv_fit_return_self)
@icontract.require(lambda estimator: _not_none(estimator), "estimator must be supplied")
@icontract.ensure(lambda result, estimator: result is estimator, "fit return must preserve estimator identity")
def lars_cv_fit_return_self(estimator: object) -> object:
    """Return the fitted LarsCV estimator identity."""
    return estimator
