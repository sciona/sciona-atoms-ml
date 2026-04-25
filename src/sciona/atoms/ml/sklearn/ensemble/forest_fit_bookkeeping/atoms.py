"""Deterministic forest fit-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from ..forest_sampling.atoms import forest_resolve_bootstrap_sample_count
from .witnesses import (
    witness_forest_fit_additional_estimator_count,
    witness_forest_fit_bootstrap_sample_count,
    witness_forest_fit_oob_update_required,
    witness_forest_fit_require_bootstrap_for_oob,
    witness_forest_fit_require_supported_oob_target_type,
)

TargetType = str


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _max_samples_spec_valid(max_samples: int | float | None) -> bool:
    if max_samples is None:
        return True
    if isinstance(max_samples, bool):
        return False
    if isinstance(max_samples, int):
        return max_samples >= 1
    if isinstance(max_samples, float):
        return bool(np.isfinite(max_samples) and 0.0 < max_samples <= 1.0)
    return False


def _bootstrap_sample_count_valid(
    result: int | None,
    bootstrap: bool,
    n_samples: int,
    max_samples: int | float | None,
) -> bool:
    if not bootstrap:
        return result is None and max_samples is None
    return bool(
        isinstance(result, int)
        and forest_resolve_bootstrap_sample_count(n_samples, max_samples) == result
    )


def _target_type_valid(target_type: object) -> bool:
    return bool(isinstance(target_type, str) and len(target_type) >= 1)


@register_atom(witness_forest_fit_bootstrap_sample_count)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(
    lambda max_samples: _max_samples_spec_valid(max_samples),
    "max_samples must be None, a positive integer, or a finite float in (0, 1]",
)
@icontract.require(
    lambda bootstrap, max_samples: bootstrap or max_samples is None,
    "max_samples can only be set when bootstrap is enabled",
)
@icontract.ensure(
    lambda result, bootstrap, n_samples, max_samples: _bootstrap_sample_count_valid(
        result, bootstrap, n_samples, max_samples
    ),
    "bootstrap sample count must match sklearn's fit-time bootstrap bookkeeping",
)
def forest_fit_bootstrap_sample_count(
    bootstrap: bool,
    n_samples: int,
    max_samples: int | float | None,
) -> int | None:
    """Return the bootstrap draw count used by forest fit, or None when bootstrapping is off."""
    if not bootstrap:
        return None
    return forest_resolve_bootstrap_sample_count(n_samples, max_samples)


@register_atom(witness_forest_fit_require_bootstrap_for_oob)
@icontract.require(
    lambda bootstrap, oob_score: bootstrap or not oob_score,
    "out-of-bag estimation is only available when bootstrap is enabled",
)
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def forest_fit_require_bootstrap_for_oob(
    bootstrap: bool,
    oob_score: bool,
) -> bool:
    """Require sklearn's bootstrap preflight rule for forest OOB estimates."""
    return True


@register_atom(witness_forest_fit_additional_estimator_count)
@icontract.require(lambda n_estimators: _positive_int(n_estimators), "n_estimators must be a positive integer")
@icontract.require(lambda existing_estimators: _nonnegative_int(existing_estimators), "existing_estimators must be a nonnegative integer")
@icontract.require(
    lambda n_estimators, existing_estimators: n_estimators >= existing_estimators,
    "n_estimators must be at least the number of existing estimators",
)
@icontract.ensure(lambda result: _nonnegative_int(result), "additional estimator count must be a nonnegative integer")
def forest_fit_additional_estimator_count(
    n_estimators: int,
    existing_estimators: int,
) -> int:
    """Compute how many new trees forest fit should grow under the current warm-start state."""
    return int(n_estimators - existing_estimators)


@register_atom(witness_forest_fit_oob_update_required)
@icontract.require(lambda n_more_estimators: _nonnegative_int(n_more_estimators), "n_more_estimators must be a nonnegative integer")
@icontract.ensure(lambda result: isinstance(result, bool), "OOB update predicate must return a boolean")
def forest_fit_oob_update_required(
    oob_score: bool,
    n_more_estimators: int,
    has_oob_score_attr: bool,
) -> bool:
    """Return whether forest fit should recompute OOB attributes after tree-growth bookkeeping."""
    return bool(oob_score and (n_more_estimators > 0 or not has_oob_score_attr))


@register_atom(witness_forest_fit_require_supported_oob_target_type)
@icontract.require(lambda target_type: _target_type_valid(target_type), "target_type must be a nonempty string")
@icontract.require(
    lambda target_type, is_classifier: target_type != "unknown" and (not is_classifier or target_type != "multiclass-multioutput"),
    "forest OOB estimates only support non-unknown targets and exclude classifier multiclass-multioutput targets",
)
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def forest_fit_require_supported_oob_target_type(
    target_type: TargetType,
    is_classifier: bool,
) -> bool:
    """Require sklearn's target-type preflight rule for forest OOB estimates."""
    return True
