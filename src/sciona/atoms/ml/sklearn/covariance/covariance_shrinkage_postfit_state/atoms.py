"""Covariance shrinkage post-fit state atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_covariance_fit_return_self,
    witness_covariance_ledoit_wolf_fit_shrinkage,
    witness_covariance_oas_fit_shrinkage,
)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _finite_unit_interval_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


@register_atom(witness_covariance_ledoit_wolf_fit_shrinkage)
@icontract.require(lambda shrinkage: _finite_unit_interval_scalar(shrinkage), "shrinkage must be a finite scalar in [0, 1]")
@icontract.ensure(lambda result, shrinkage: _finite_unit_interval_scalar(result) and float(result) == float(shrinkage), "result must expose LedoitWolf.shrinkage_ unchanged")
def covariance_ledoit_wolf_fit_shrinkage(
    shrinkage: float,
) -> float:
    """Expose the fitted `shrinkage_` scalar from LedoitWolf.fit."""
    return float(shrinkage)


@register_atom(witness_covariance_oas_fit_shrinkage)
@icontract.require(lambda shrinkage: _finite_unit_interval_scalar(shrinkage), "shrinkage must be a finite scalar in [0, 1]")
@icontract.ensure(lambda result, shrinkage: _finite_unit_interval_scalar(result) and float(result) == float(shrinkage), "result must expose OAS.shrinkage_ unchanged")
def covariance_oas_fit_shrinkage(
    shrinkage: float,
) -> float:
    """Expose the fitted `shrinkage_` scalar from OAS.fit."""
    return float(shrinkage)


@register_atom(witness_covariance_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(lambda result, estimator_token: isinstance(result, str) and result == estimator_token, "result must return the estimator token unchanged")
def covariance_fit_return_self(
    estimator_token: str,
) -> str:
    """Model covariance-estimator `fit` methods returning the fitted estimator itself."""
    return estimator_token
