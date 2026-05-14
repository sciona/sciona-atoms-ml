"""Ghost witnesses for sklearn RANSAC callback-orchestration atoms."""

from __future__ import annotations

from collections.abc import Mapping


def witness_ransac_nonrouting_estimator_params(sample_weight: object) -> Mapping[str, object]:
    """Describe RANSAC non-routing estimator parameter containers."""
    return {"sample_weight": sample_weight}


def witness_ransac_skip_limit_exceeded(
    n_skips_no_inliers: int,
    n_skips_invalid_data: int,
    n_skips_invalid_model: int,
    max_skips: int,
) -> bool:
    """Describe the RANSAC max-skips guard."""
    return (n_skips_no_inliers + n_skips_invalid_data + n_skips_invalid_model) > max_skips


def witness_ransac_trial_subset_payload(X: object, y: object, subset_idxs: object) -> Mapping[str, object]:
    """Describe one RANSAC trial subset payload."""
    return {"X": X, "y": y, "subset_idxs": subset_idxs}


def witness_ransac_estimator_fit_callback_payload(
    X_subset: object,
    y_subset: object,
    fit_params_subset: Mapping[str, object],
) -> Mapping[str, object]:
    """Describe the estimator.fit callback payload for one RANSAC trial."""
    return {"X_subset": X_subset, "y_subset": y_subset, "fit_params_subset": fit_params_subset}


def witness_ransac_inlier_subset_payload(X: object, y: object, sample_idxs: object, inlier_mask_subset: object) -> Mapping[str, object]:
    """Describe the accepted RANSAC inlier subset payload."""
    return {"X": X, "y": y, "sample_idxs": sample_idxs, "inlier_mask_subset": inlier_mask_subset}


def witness_ransac_score_callback_payload(
    X_inlier_subset: object,
    y_inlier_subset: object,
    score_params_inlier_subset: Mapping[str, object],
) -> Mapping[str, object]:
    """Describe the estimator.score callback payload for a RANSAC consensus."""
    return {
        "X_inlier_subset": X_inlier_subset,
        "y_inlier_subset": y_inlier_subset,
        "score_params_inlier_subset": score_params_inlier_subset,
    }


def witness_ransac_no_consensus_failure_message(skip_limit_exceeded: bool) -> str:
    """Describe the no-consensus RANSAC failure message choice."""
    return str(skip_limit_exceeded)


def witness_ransac_final_refit_callback_payload(
    X_inlier_best: object,
    y_inlier_best: object,
    fit_params_best_idxs_subset: Mapping[str, object],
) -> Mapping[str, object]:
    """Describe the final estimator.fit callback payload on best inliers."""
    return {
        "X_inlier_best": X_inlier_best,
        "y_inlier_best": y_inlier_best,
        "fit_params_best_idxs_subset": fit_params_best_idxs_subset,
    }
