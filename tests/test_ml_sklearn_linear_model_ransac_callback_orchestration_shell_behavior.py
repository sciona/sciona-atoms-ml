from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_ransac_callback_orchestration_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_callback_orchestration_shell import (
        ransac_estimator_fit_callback_payload,
        ransac_final_refit_callback_payload,
        ransac_inlier_subset_payload,
        ransac_no_consensus_failure_message,
        ransac_nonrouting_estimator_params,
        ransac_score_callback_payload,
        ransac_skip_limit_exceeded,
        ransac_trial_subset_payload,
    )

    assert callable(ransac_nonrouting_estimator_params)
    assert callable(ransac_skip_limit_exceeded)
    assert callable(ransac_trial_subset_payload)
    assert callable(ransac_estimator_fit_callback_payload)
    assert callable(ransac_inlier_subset_payload)
    assert callable(ransac_score_callback_payload)
    assert callable(ransac_no_consensus_failure_message)
    assert callable(ransac_final_refit_callback_payload)


def test_ransac_nonrouting_estimator_params_match_fallback_shape() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_callback_orchestration_shell import ransac_nonrouting_estimator_params

    assert ransac_nonrouting_estimator_params(None) == {"fit": {}, "predict": {}, "score": {}}

    sample_weight = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    routed = ransac_nonrouting_estimator_params(sample_weight)
    assert routed["fit"]["sample_weight"] is sample_weight
    assert routed["predict"] == {}
    assert routed["score"] == {}


def test_ransac_skip_limit_exceeded_matches_source_guard() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_callback_orchestration_shell import ransac_skip_limit_exceeded

    assert ransac_skip_limit_exceeded(1, 2, 3, 6) is False
    assert ransac_skip_limit_exceeded(1, 2, 4, 6) is True


def test_ransac_trial_subset_payload_matches_source_slicing() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_callback_orchestration_shell import ransac_trial_subset_payload

    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    subset_idxs = np.array([2, 0], dtype=np.int64)

    payload = ransac_trial_subset_payload(X, y, subset_idxs)

    assert np.array_equal(payload["subset_idxs"], subset_idxs)
    assert np.array_equal(payload["X_subset"], X[subset_idxs])
    assert np.array_equal(payload["y_subset"], y[subset_idxs])


def test_ransac_fit_score_and_final_refit_payloads_preserve_callback_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_callback_orchestration_shell import (
        ransac_estimator_fit_callback_payload,
        ransac_final_refit_callback_payload,
        ransac_score_callback_payload,
    )

    X_subset = np.array([[1.0], [2.0]], dtype=np.float64)
    y_subset = np.array([1.0, 2.0], dtype=np.float64)
    params = {"sample_weight": np.array([0.5, 1.0], dtype=np.float64)}

    fit_payload = ransac_estimator_fit_callback_payload(X_subset, y_subset, params)
    score_payload = ransac_score_callback_payload(X_subset, y_subset, params)
    final_payload = ransac_final_refit_callback_payload(X_subset, y_subset, params)

    assert fit_payload["X_subset"] is X_subset
    assert fit_payload["y_subset"] is y_subset
    assert fit_payload["fit_params_subset"] is params
    assert score_payload["X_inlier_subset"] is X_subset
    assert score_payload["y_inlier_subset"] is y_subset
    assert score_payload["score_params_inlier_subset"] is params
    assert final_payload["X_inlier_best"] is X_subset
    assert final_payload["y_inlier_best"] is y_subset
    assert final_payload["fit_params_best_idxs_subset"] is params


def test_ransac_inlier_subset_payload_matches_source_index_extraction() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_callback_orchestration_shell import ransac_inlier_subset_payload

    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    sample_idxs = np.arange(X.shape[0])
    inlier_mask_subset = np.array([True, False, True, False], dtype=np.bool_)

    payload = ransac_inlier_subset_payload(X, y, sample_idxs, inlier_mask_subset)

    assert np.array_equal(payload["inlier_idxs_subset"], np.array([0, 2]))
    assert np.array_equal(payload["X_inlier_subset"], X[[0, 2]])
    assert np.array_equal(payload["y_inlier_subset"], y[[0, 2]])


def test_ransac_no_consensus_failure_message_matches_sklearn_wording() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_callback_orchestration_shell import ransac_no_consensus_failure_message

    max_skips_message = ransac_no_consensus_failure_message(True)
    all_trials_message = ransac_no_consensus_failure_message(False)

    assert max_skips_message == (
        "RANSAC skipped more iterations than `max_skips` without"
        " finding a valid consensus set. Iterations were skipped"
        " because each randomly chosen sub-sample failed the"
        " passing criteria. See estimator attributes for"
        " diagnostics (n_skips*)."
    )
    assert all_trials_message == (
        "RANSAC could not find a valid consensus set. All"
        " `max_trials` iterations were skipped because each"
        " randomly chosen sub-sample failed the passing criteria."
        " See estimator attributes for diagnostics (n_skips*)."
    )


def test_ransac_callback_orchestration_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_callback_orchestration_shell import (
        ransac_estimator_fit_callback_payload,
        ransac_inlier_subset_payload,
        ransac_no_consensus_failure_message,
        ransac_skip_limit_exceeded,
        ransac_trial_subset_payload,
    )

    with pytest.raises(ViolationError):
        ransac_skip_limit_exceeded(-1, 0, 0, 0)

    with pytest.raises(ViolationError):
        ransac_trial_subset_payload(np.ones((2, 1)), np.ones(3), np.array([0]))

    with pytest.raises(ViolationError):
        ransac_trial_subset_payload(np.ones((2, 1)), np.ones(2), np.array([2]))

    with pytest.raises(ViolationError):
        ransac_estimator_fit_callback_payload(object(), object(), None)

    with pytest.raises(ViolationError):
        ransac_inlier_subset_payload(np.ones((2, 1)), np.ones(2), np.arange(2), np.array([True]))

    with pytest.raises(ViolationError):
        ransac_no_consensus_failure_message(1)
