"""Sklearn RANSAC fit prelude and termination atoms."""

from __future__ import annotations

from numbers import Integral, Real

import icontract
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_ransac_min_samples_guard_payload,
    witness_ransac_min_samples_value,
    witness_ransac_stop_condition_reached,
    witness_ransac_valid_consensus_skip_warning_payload,
)

_MIN_SAMPLES_GUARD_KEYS = {"too_large", "message"}
_WARNING_PAYLOAD_KEYS = {"should_warn", "category", "message"}
_EXPLICIT_MIN_SAMPLES_MESSAGE = (
    "`min_samples` needs to be explicitly set when estimator is not a LinearRegression."
)
_VALID_CONSENSUS_MAX_SKIPS_MESSAGE = (
    "RANSAC found a valid consensus set but exited"
    " early due to skipping more iterations than"
    " `max_skips`. See estimator attributes for"
    " diagnostics (n_skips*)."
)


def _positive_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0)


def _nonnegative_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 0)


def _min_samples_valid(min_samples: object, estimator_is_linear_regression: bool) -> bool:
    if not isinstance(estimator_is_linear_regression, bool):
        return False
    if min_samples is None:
        return estimator_is_linear_regression
    return bool(
        isinstance(min_samples, Real)
        and not isinstance(min_samples, bool)
        and np.isfinite(float(min_samples))
        and float(min_samples) > 0.0
    )


def _score_valid(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(float(value)))


def _stop_score_valid(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and not np.isnan(float(value)))


def _min_samples_result_valid(result: int, min_samples: object, n_samples: int, n_features: int) -> bool:
    if min_samples is None:
        expected = int(n_features) + 1
    elif 0 < min_samples < 1:
        expected = int(np.ceil(float(min_samples) * int(n_samples)))
    else:
        expected = int(min_samples)
    return bool(result == expected and isinstance(result, int) and result > 0)


def _min_samples_guard_valid(result: dict[str, object], resolved_min_samples: int, n_samples: int) -> bool:
    too_large = int(resolved_min_samples) > int(n_samples)
    message = "`min_samples` may not be larger than number of samples: n_samples = %d." % int(n_samples)
    return bool(
        set(result) == _MIN_SAMPLES_GUARD_KEYS
        and result["too_large"] is too_large
        and result["message"] == (message if too_large else None)
    )


def _warning_payload_valid(result: dict[str, object], total_skips: int, max_skips: int) -> bool:
    should_warn = int(total_skips) > int(max_skips)
    return bool(
        set(result) == _WARNING_PAYLOAD_KEYS
        and result["should_warn"] is should_warn
        and result["category"] is (ConvergenceWarning if should_warn else None)
        and result["message"] == (_VALID_CONSENSUS_MAX_SKIPS_MESSAGE if should_warn else None)
    )


@register_atom(witness_ransac_min_samples_value)
@icontract.require(lambda n_samples: _positive_integer(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda n_features: _positive_integer(n_features), "n_features must be a positive integer")
@icontract.require(
    lambda min_samples, estimator_is_linear_regression: _min_samples_valid(min_samples, estimator_is_linear_regression),
    _EXPLICIT_MIN_SAMPLES_MESSAGE,
)
@icontract.ensure(
    lambda result, min_samples, n_samples, n_features: _min_samples_result_valid(result, min_samples, n_samples, n_features),
    "resolved min_samples must match RANSACRegressor.fit source branches",
)
def ransac_min_samples_value(
    min_samples: object,
    n_samples: int,
    n_features: int,
    *,
    estimator_is_linear_regression: bool,
) -> int:
    """Resolve RANSACRegressor.fit min_samples before the too-large guard."""
    if min_samples is None:
        return int(n_features) + 1
    if 0 < min_samples < 1:
        return int(np.ceil(float(min_samples) * int(n_samples)))
    return int(min_samples)


@register_atom(witness_ransac_min_samples_guard_payload)
@icontract.require(lambda resolved_min_samples: _positive_integer(resolved_min_samples), "resolved_min_samples must be positive")
@icontract.require(lambda n_samples: _positive_integer(n_samples), "n_samples must be a positive integer")
@icontract.ensure(
    lambda result, resolved_min_samples, n_samples: _min_samples_guard_valid(result, resolved_min_samples, n_samples),
    "min_samples guard payload must match source too-large branch",
)
def ransac_min_samples_guard_payload(resolved_min_samples: int, n_samples: int) -> dict[str, object]:
    """Return the min_samples-too-large guard and source error message."""
    too_large = int(resolved_min_samples) > int(n_samples)
    return {
        "too_large": too_large,
        "message": "`min_samples` may not be larger than number of samples: n_samples = %d." % int(n_samples)
        if too_large
        else None,
    }


@register_atom(witness_ransac_stop_condition_reached)
@icontract.require(lambda n_inliers_best: _nonnegative_integer(n_inliers_best), "n_inliers_best must be nonnegative")
@icontract.require(lambda stop_n_inliers: _nonnegative_integer(stop_n_inliers), "stop_n_inliers must be nonnegative")
@icontract.require(lambda score_best: _score_valid(score_best), "score_best must be finite")
@icontract.require(lambda stop_score: _stop_score_valid(stop_score), "stop_score must be numeric and not NaN")
@icontract.ensure(
    lambda result, n_inliers_best, score_best, stop_n_inliers, stop_score: result
    is (int(n_inliers_best) >= int(stop_n_inliers) or float(score_best) >= float(stop_score)),
    "stop guard must match RANSACRegressor.fit accepted-consensus branch",
)
def ransac_stop_condition_reached(
    n_inliers_best: int,
    score_best: float,
    stop_n_inliers: int,
    stop_score: float,
) -> bool:
    """Return whether accepted consensus is sufficient to stop the RANSAC loop."""
    return int(n_inliers_best) >= int(stop_n_inliers) or float(score_best) >= float(stop_score)


@register_atom(witness_ransac_valid_consensus_skip_warning_payload)
@icontract.require(lambda total_skips: _nonnegative_integer(total_skips), "total_skips must be nonnegative")
@icontract.require(lambda max_skips: _nonnegative_integer(max_skips), "max_skips must be nonnegative")
@icontract.ensure(
    lambda result, total_skips, max_skips: _warning_payload_valid(result, total_skips, max_skips),
    "warning payload must match valid-consensus max-skips source branch",
)
def ransac_valid_consensus_skip_warning_payload(total_skips: int, max_skips: int) -> dict[str, object]:
    """Return the warning payload for valid consensus with too many skipped trials."""
    should_warn = int(total_skips) > int(max_skips)
    return {
        "should_warn": should_warn,
        "category": ConvergenceWarning if should_warn else None,
        "message": _VALID_CONSENSUS_MAX_SKIPS_MESSAGE if should_warn else None,
    }
