"""Ghost witnesses for sklearn RANSAC fit prelude and termination atoms."""

from __future__ import annotations

import numpy as np
from sklearn.exceptions import ConvergenceWarning


_VALID_CONSENSUS_MAX_SKIPS_MESSAGE = (
    "RANSAC found a valid consensus set but exited"
    " early due to skipping more iterations than"
    " `max_skips`. See estimator attributes for"
    " diagnostics (n_skips*)."
)


def witness_ransac_min_samples_value(
    min_samples: object,
    n_samples: int,
    n_features: int,
    estimator_is_linear_regression: bool,
) -> int:
    """Describe RANSACRegressor.fit min_samples resolution."""
    del estimator_is_linear_regression
    if min_samples is None:
        return int(n_features) + 1
    if 0 < min_samples < 1:
        return int(np.ceil(float(min_samples) * int(n_samples)))
    return int(min_samples)


def witness_ransac_min_samples_guard_payload(resolved_min_samples: int, n_samples: int) -> dict[str, object]:
    """Describe the min_samples-too-large guard payload."""
    too_large = int(resolved_min_samples) > int(n_samples)
    return {
        "too_large": too_large,
        "message": "`min_samples` may not be larger than number of samples: n_samples = %d." % int(n_samples)
        if too_large
        else None,
    }


def witness_ransac_stop_condition_reached(
    n_inliers_best: int,
    score_best: float,
    stop_n_inliers: int,
    stop_score: float,
) -> bool:
    """Describe the accepted-consensus stop guard."""
    return bool(int(n_inliers_best) >= int(stop_n_inliers) or float(score_best) >= float(stop_score))


def witness_ransac_valid_consensus_skip_warning_payload(total_skips: int, max_skips: int) -> dict[str, object]:
    """Describe the valid-consensus max-skips warning payload."""
    should_warn = int(total_skips) > int(max_skips)
    return {
        "should_warn": should_warn,
        "category": ConvergenceWarning if should_warn else None,
        "message": _VALID_CONSENSUS_MAX_SKIPS_MESSAGE if should_warn else None,
    }
