"""Bagging fit-shell bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bagging_additional_estimator_count,
    witness_bagging_fit_require_bootstrap_for_oob,
    witness_bagging_fit_require_no_warm_start_with_oob,
    witness_bagging_resolve_max_features,
    witness_bagging_resolve_max_samples,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _count_spec_valid(value: int | float) -> bool:
    return bool(
        (isinstance(value, int) and not isinstance(value, bool) and value >= 1)
        or (
            isinstance(value, float)
            and np.isfinite(value)
            and value > 0.0
        )
    )


def _override_spec_valid(value: int | float | None) -> bool:
    return bool(value is None or _count_spec_valid(value))


@register_atom(witness_bagging_resolve_max_samples)
@icontract.require(lambda max_samples_override: _override_spec_valid(max_samples_override), "max_samples_override must be None, a positive integer, or a positive finite float")
@icontract.require(lambda configured_max_samples: _count_spec_valid(configured_max_samples), "configured_max_samples must be a positive integer or positive finite float")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(
    lambda max_samples_override, configured_max_samples, n_samples: (
        int((configured_max_samples if max_samples_override is None else max_samples_override) * n_samples)
        if isinstance(configured_max_samples if max_samples_override is None else max_samples_override, float)
        else int(configured_max_samples if max_samples_override is None else max_samples_override)
    ) <= n_samples,
    "resolved max_samples must be <= n_samples",
)
@icontract.ensure(lambda result, n_samples: isinstance(result, int) and 1 <= result <= n_samples, "resolved max_samples must be an integer in [1, n_samples]")
def bagging_resolve_max_samples(
    max_samples_override: int | float | None,
    configured_max_samples: int | float,
    n_samples: int,
) -> int:
    """Resolve sklearn bagging's validated integer sample draw count."""
    value = configured_max_samples if max_samples_override is None else max_samples_override
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return int(float(value) * n_samples)


@register_atom(witness_bagging_resolve_max_features)
@icontract.require(lambda configured_max_features: _count_spec_valid(configured_max_features), "configured_max_features must be a positive integer or positive finite float")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(
    lambda configured_max_features, n_features: (
        int(configured_max_features * n_features) if isinstance(configured_max_features, float) else int(configured_max_features)
    ) <= n_features,
    "resolved max_features must be <= n_features before clamping",
)
@icontract.ensure(lambda result, n_features: isinstance(result, int) and 1 <= result <= n_features, "resolved max_features must be an integer in [1, n_features]")
def bagging_resolve_max_features(
    configured_max_features: int | float,
    n_features: int,
) -> int:
    """Resolve sklearn bagging's validated integer feature draw count."""
    if isinstance(configured_max_features, int) and not isinstance(configured_max_features, bool):
        resolved = int(configured_max_features)
    else:
        resolved = int(float(configured_max_features) * n_features)
    return max(1, resolved)


@register_atom(witness_bagging_fit_require_bootstrap_for_oob)
@icontract.require(
    lambda bootstrap, oob_score: bootstrap or not oob_score,
    "out-of-bag estimation is only available when bootstrap is enabled",
)
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def bagging_fit_require_bootstrap_for_oob(
    bootstrap: bool,
    oob_score: bool,
) -> bool:
    """Require sklearn's bootstrap preflight rule for out-of-bag estimation."""
    return True


@register_atom(witness_bagging_fit_require_no_warm_start_with_oob)
@icontract.require(
    lambda warm_start, oob_score: (not warm_start) or (not oob_score),
    "out-of-bag estimation is only available when warm_start is disabled",
)
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def bagging_fit_require_no_warm_start_with_oob(
    warm_start: bool,
    oob_score: bool,
) -> bool:
    """Require sklearn's no-warm-start preflight rule for out-of-bag estimation."""
    return True


@register_atom(witness_bagging_additional_estimator_count)
@icontract.require(lambda n_estimators: _positive_int(n_estimators), "n_estimators must be a positive integer")
@icontract.require(lambda existing_estimators: isinstance(existing_estimators, int) and not isinstance(existing_estimators, bool) and existing_estimators >= 0, "existing_estimators must be a nonnegative integer")
@icontract.require(
    lambda n_estimators, existing_estimators: n_estimators >= existing_estimators,
    "n_estimators must be at least the number of existing estimators",
)
@icontract.ensure(lambda result: isinstance(result, int) and result >= 0, "additional estimator count must be a nonnegative integer")
def bagging_additional_estimator_count(
    n_estimators: int,
    existing_estimators: int,
) -> int:
    """Compute how many new estimators sklearn bagging fit should build."""
    return int(n_estimators - existing_estimators)
