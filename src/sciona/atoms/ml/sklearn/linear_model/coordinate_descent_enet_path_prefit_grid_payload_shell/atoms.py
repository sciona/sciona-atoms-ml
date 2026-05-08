"""Sklearn coordinate-descent enet_path pre-fit/grid payload atoms adapted from scikit-learn."""

from __future__ import annotations

from numbers import Integral, Real

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_alpha_grid_18_kwargs,
    witness_cd_enet_path_prefit_18_kwargs,
)


def _enet_path(value: object) -> bool:
    return value == "enet_path"


def _positive_integral(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 1


def _positive_real(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(isinstance(value, Real) and numeric > 0.0)


@register_atom(witness_cd_enet_path_prefit_18_kwargs)
@icontract.require(lambda path_helper: _enet_path(path_helper), "path_helper must be enet_path")
@icontract.ensure(
    lambda result: result == {"fit_intercept": False, "copy": False, "check_gram": True},
    "pre-fit kwargs must match sklearn 1.8 enet_path",
)
def cd_enet_path_prefit_18_kwargs(path_helper: str) -> dict[str, bool]:
    """Return the fixed _pre_fit keyword payload used by sklearn 1.8 enet_path."""
    del path_helper
    return {"fit_intercept": False, "copy": False, "check_gram": True}


@register_atom(witness_cd_enet_path_alpha_grid_18_kwargs)
@icontract.require(lambda path_helper: _enet_path(path_helper), "path_helper must be enet_path")
@icontract.require(lambda eps: _positive_real(eps), "eps must be positive")
@icontract.require(lambda n_alphas: _positive_integral(n_alphas), "n_alphas must be a positive integer")
@icontract.ensure(
    lambda result, Xy, l1_ratio, eps, n_alphas: isinstance(result, dict)
    and set(result) == {"Xy", "l1_ratio", "fit_intercept", "eps", "n_alphas"}
    and result["Xy"] is Xy
    and result["l1_ratio"] is l1_ratio
    and result["fit_intercept"] is False
    and result["eps"] is eps
    and result["n_alphas"] is n_alphas,
    "alpha-grid kwargs must match sklearn 1.8 enet_path",
)
def cd_enet_path_alpha_grid_18_kwargs(
    path_helper: str,
    Xy: object,
    l1_ratio: object,
    eps: Real,
    n_alphas: Integral,
) -> dict[str, object]:
    """Return the _alpha_grid keyword payload used by sklearn 1.8 enet_path."""
    del path_helper
    return {
        "Xy": Xy,
        "l1_ratio": l1_ratio,
        "fit_intercept": False,
        "eps": eps,
        "n_alphas": n_alphas,
    }
