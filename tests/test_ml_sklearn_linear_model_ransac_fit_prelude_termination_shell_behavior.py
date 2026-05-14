from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.exceptions import ConvergenceWarning


def test_ransac_fit_prelude_termination_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_fit_prelude_termination_shell import (
        ransac_min_samples_guard_payload,
        ransac_min_samples_value,
        ransac_stop_condition_reached,
        ransac_valid_consensus_skip_warning_payload,
    )

    assert callable(ransac_min_samples_value)
    assert callable(ransac_min_samples_guard_payload)
    assert callable(ransac_stop_condition_reached)
    assert callable(ransac_valid_consensus_skip_warning_payload)


def test_ransac_min_samples_value_matches_source_branches() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_fit_prelude_termination_shell import ransac_min_samples_value

    assert ransac_min_samples_value(None, 10, 3, estimator_is_linear_regression=True) == 4
    assert ransac_min_samples_value(0.25, 10, 3, estimator_is_linear_regression=False) == 3
    assert ransac_min_samples_value(4, 10, 3, estimator_is_linear_regression=False) == 4
    assert ransac_min_samples_value(np.float64(0.51), 10, 3, estimator_is_linear_regression=False) == 6


def test_ransac_min_samples_guard_payload_matches_source_message() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_fit_prelude_termination_shell import ransac_min_samples_guard_payload

    assert ransac_min_samples_guard_payload(4, 10) == {"too_large": False, "message": None}
    assert ransac_min_samples_guard_payload(11, 10) == {
        "too_large": True,
        "message": "`min_samples` may not be larger than number of samples: n_samples = 10.",
    }


def test_ransac_stop_condition_reached_matches_accepted_consensus_guard() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_fit_prelude_termination_shell import ransac_stop_condition_reached

    assert ransac_stop_condition_reached(5, 0.2, 5, 1.0) is True
    assert ransac_stop_condition_reached(4, 1.1, 5, 1.0) is True
    assert ransac_stop_condition_reached(4, 0.9, 5, 1.0) is False
    assert ransac_stop_condition_reached(4, 0.9, 5, float("inf")) is False


def test_ransac_valid_consensus_skip_warning_payload_matches_source_warning() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_fit_prelude_termination_shell import (
        ransac_valid_consensus_skip_warning_payload,
    )

    quiet = ransac_valid_consensus_skip_warning_payload(3, 3)
    warning = ransac_valid_consensus_skip_warning_payload(4, 3)

    assert quiet == {"should_warn": False, "category": None, "message": None}
    assert warning["should_warn"] is True
    assert warning["category"] is ConvergenceWarning
    assert warning["message"] == (
        "RANSAC found a valid consensus set but exited"
        " early due to skipping more iterations than"
        " `max_skips`. See estimator attributes for"
        " diagnostics (n_skips*)."
    )


def test_ransac_fit_prelude_termination_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_fit_prelude_termination_shell import (
        ransac_min_samples_guard_payload,
        ransac_min_samples_value,
        ransac_stop_condition_reached,
        ransac_valid_consensus_skip_warning_payload,
    )

    with pytest.raises(ViolationError):
        ransac_min_samples_value(None, 10, 3, estimator_is_linear_regression=False)

    with pytest.raises(ViolationError):
        ransac_min_samples_value(0, 10, 3, estimator_is_linear_regression=True)

    with pytest.raises(ViolationError):
        ransac_min_samples_guard_payload(0, 10)

    with pytest.raises(ViolationError):
        ransac_stop_condition_reached(-1, 0.0, 1, 1.0)

    with pytest.raises(ViolationError):
        ransac_stop_condition_reached(1, float("nan"), 1, 1.0)

    with pytest.raises(ViolationError):
        ransac_valid_consensus_skip_warning_payload(-1, 0)
