"""Sklearn RANSAC callback-orchestration atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_ransac_estimator_fit_callback_payload,
    witness_ransac_final_refit_callback_payload,
    witness_ransac_inlier_subset_payload,
    witness_ransac_no_consensus_failure_message,
    witness_ransac_nonrouting_estimator_params,
    witness_ransac_score_callback_payload,
    witness_ransac_skip_limit_exceeded,
    witness_ransac_trial_subset_payload,
)

_NONROUTING_KEYS = {"fit", "predict", "score"}
_TRIAL_SUBSET_KEYS = {"subset_idxs", "X_subset", "y_subset"}
_FIT_PAYLOAD_KEYS = {"X_subset", "y_subset", "fit_params_subset"}
_INLIER_SUBSET_KEYS = {"inlier_idxs_subset", "X_inlier_subset", "y_inlier_subset"}
_SCORE_PAYLOAD_KEYS = {"X_inlier_subset", "y_inlier_subset", "score_params_inlier_subset"}
_FINAL_REFIT_KEYS = {"X_inlier_best", "y_inlier_best", "fit_params_best_idxs_subset"}
_MAX_SKIPS_NO_VALID_CONSENSUS_MESSAGE = (
    "RANSAC skipped more iterations than `max_skips` without"
    " finding a valid consensus set. Iterations were skipped"
    " because each randomly chosen sub-sample failed the"
    " passing criteria. See estimator attributes for"
    " diagnostics (n_skips*)."
)
_ALL_TRIALS_SKIPPED_MESSAGE = (
    "RANSAC could not find a valid consensus set. All"
    " `max_trials` iterations were skipped because each"
    " randomly chosen sub-sample failed the passing criteria."
    " See estimator attributes for diagnostics (n_skips*)."
)


def _count_valid(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _array_with_samples(values: object) -> bool:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim >= 1 and array.shape[0] >= 1)


def _aligned_samples(X: object, y: object) -> bool:
    return bool(_array_with_samples(X) and _array_with_samples(y) and np.asarray(X).shape[0] == np.asarray(y).shape[0])


def _indices_valid(indices: object, n_samples: int) -> bool:
    try:
        values = np.asarray(indices, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(values >= 0) and np.all(values < n_samples))


def _mask_valid(mask: object, n_samples: int) -> bool:
    try:
        values = np.asarray(mask)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] == n_samples and values.dtype == np.bool_)


def _mapping_valid(params: Mapping[str, object]) -> bool:
    return isinstance(params, Mapping)


def _nonrouting_params_valid(result: Mapping[str, Mapping[str, object]], sample_weight: object) -> bool:
    fit_payload = result.get("fit")
    return bool(
        set(result) == _NONROUTING_KEYS
        and isinstance(fit_payload, dict)
        and isinstance(result.get("predict"), dict)
        and isinstance(result.get("score"), dict)
        and (fit_payload == {} if sample_weight is None else fit_payload.get("sample_weight") is sample_weight and len(fit_payload) == 1)
        and result["predict"] == {}
        and result["score"] == {}
    )


def _trial_subset_valid(result: Mapping[str, object], X: object, y: object, subset_idxs: object) -> bool:
    indices = np.asarray(subset_idxs, dtype=np.int64)
    return bool(
        set(result) == _TRIAL_SUBSET_KEYS
        and np.array_equal(result["subset_idxs"], indices)
        and np.array_equal(result["X_subset"], np.asarray(X)[indices])
        and np.array_equal(result["y_subset"], np.asarray(y)[indices])
    )


def _fit_payload_valid(result: Mapping[str, object], X_subset: object, y_subset: object, fit_params_subset: Mapping[str, object]) -> bool:
    return bool(
        set(result) == _FIT_PAYLOAD_KEYS
        and result["X_subset"] is X_subset
        and result["y_subset"] is y_subset
        and result["fit_params_subset"] is fit_params_subset
    )


def _inlier_subset_valid(result: Mapping[str, object], X: object, y: object, sample_idxs: object, inlier_mask_subset: object) -> bool:
    sample_indices = np.asarray(sample_idxs)
    mask = np.asarray(inlier_mask_subset)
    inlier_indices = sample_indices[mask]
    return bool(
        set(result) == _INLIER_SUBSET_KEYS
        and np.array_equal(result["inlier_idxs_subset"], inlier_indices)
        and np.array_equal(result["X_inlier_subset"], np.asarray(X)[inlier_indices])
        and np.array_equal(result["y_inlier_subset"], np.asarray(y)[inlier_indices])
    )


def _score_payload_valid(
    result: Mapping[str, object],
    X_inlier_subset: object,
    y_inlier_subset: object,
    score_params_inlier_subset: Mapping[str, object],
) -> bool:
    return bool(
        set(result) == _SCORE_PAYLOAD_KEYS
        and result["X_inlier_subset"] is X_inlier_subset
        and result["y_inlier_subset"] is y_inlier_subset
        and result["score_params_inlier_subset"] is score_params_inlier_subset
    )


def _final_refit_payload_valid(
    result: Mapping[str, object],
    X_inlier_best: object,
    y_inlier_best: object,
    fit_params_best_idxs_subset: Mapping[str, object],
) -> bool:
    return bool(
        set(result) == _FINAL_REFIT_KEYS
        and result["X_inlier_best"] is X_inlier_best
        and result["y_inlier_best"] is y_inlier_best
        and result["fit_params_best_idxs_subset"] is fit_params_best_idxs_subset
    )


@register_atom(witness_ransac_nonrouting_estimator_params)
@icontract.ensure(
    lambda result, sample_weight: _nonrouting_params_valid(result, sample_weight),
    "non-routing estimator params must match RANSAC fallback containers",
)
def ransac_nonrouting_estimator_params(sample_weight: object) -> dict[str, dict[str, object]]:
    """Return RANSAC's non-routing estimator parameter containers."""
    fit_params = {} if sample_weight is None else {"sample_weight": sample_weight}
    return {"fit": fit_params, "predict": {}, "score": {}}


@register_atom(witness_ransac_skip_limit_exceeded)
@icontract.require(lambda n_skips_no_inliers: _count_valid(n_skips_no_inliers), "n_skips_no_inliers must be nonnegative")
@icontract.require(lambda n_skips_invalid_data: _count_valid(n_skips_invalid_data), "n_skips_invalid_data must be nonnegative")
@icontract.require(lambda n_skips_invalid_model: _count_valid(n_skips_invalid_model), "n_skips_invalid_model must be nonnegative")
@icontract.require(lambda max_skips: _count_valid(max_skips), "max_skips must be nonnegative")
@icontract.ensure(
    lambda result, n_skips_no_inliers, n_skips_invalid_data, n_skips_invalid_model, max_skips: result
    is ((n_skips_no_inliers + n_skips_invalid_data + n_skips_invalid_model) > max_skips),
    "RANSAC skip-limit guard must compare total skips against max_skips",
)
def ransac_skip_limit_exceeded(
    n_skips_no_inliers: int,
    n_skips_invalid_data: int,
    n_skips_invalid_model: int,
    max_skips: int,
) -> bool:
    """Return whether RANSAC has skipped more iterations than max_skips."""
    return (int(n_skips_no_inliers) + int(n_skips_invalid_data) + int(n_skips_invalid_model)) > int(max_skips)


@register_atom(witness_ransac_trial_subset_payload)
@icontract.require(lambda X, y: _aligned_samples(X, y), "X and y must have matching samples")
@icontract.require(lambda subset_idxs, X: _indices_valid(subset_idxs, np.asarray(X).shape[0]), "subset indices must select existing samples")
@icontract.ensure(
    lambda result, X, y, subset_idxs: _trial_subset_valid(result, X, y, subset_idxs),
    "trial subset payload must match RANSAC subset slicing",
)
def ransac_trial_subset_payload(X: object, y: object, subset_idxs: object) -> dict[str, object]:
    """Return the X/y subset payload for one RANSAC trial."""
    indices = np.asarray(subset_idxs, dtype=np.int64)
    return {
        "subset_idxs": indices,
        "X_subset": np.asarray(X)[indices],
        "y_subset": np.asarray(y)[indices],
    }


@register_atom(witness_ransac_estimator_fit_callback_payload)
@icontract.require(lambda X_subset: X_subset is not None, "X_subset must be provided")
@icontract.require(lambda y_subset: y_subset is not None, "y_subset must be provided")
@icontract.require(lambda fit_params_subset: _mapping_valid(fit_params_subset), "fit_params_subset must be a mapping")
@icontract.ensure(
    lambda result, X_subset, y_subset, fit_params_subset: _fit_payload_valid(result, X_subset, y_subset, fit_params_subset),
    "fit callback payload must preserve trial subset and sliced fit params",
)
def ransac_estimator_fit_callback_payload(
    X_subset: object,
    y_subset: object,
    fit_params_subset: Mapping[str, object],
) -> dict[str, object]:
    """Return the estimator.fit callback payload for one RANSAC trial."""
    return {"X_subset": X_subset, "y_subset": y_subset, "fit_params_subset": fit_params_subset}


@register_atom(witness_ransac_inlier_subset_payload)
@icontract.require(lambda X, y: _aligned_samples(X, y), "X and y must have matching samples")
@icontract.require(lambda sample_idxs, X: _indices_valid(sample_idxs, np.asarray(X).shape[0]), "sample indices must select existing samples")
@icontract.require(lambda inlier_mask_subset, sample_idxs: _mask_valid(inlier_mask_subset, np.asarray(sample_idxs).shape[0]), "inlier mask must align with sample indices")
@icontract.ensure(
    lambda result, X, y, sample_idxs, inlier_mask_subset: _inlier_subset_valid(result, X, y, sample_idxs, inlier_mask_subset),
    "inlier subset payload must match RANSAC inlier extraction",
)
def ransac_inlier_subset_payload(
    X: object,
    y: object,
    sample_idxs: object,
    inlier_mask_subset: object,
) -> dict[str, object]:
    """Return accepted inlier indices and X/y subsets for a RANSAC consensus."""
    indices = np.asarray(sample_idxs)
    mask = np.asarray(inlier_mask_subset)
    inlier_indices = indices[mask]
    return {
        "inlier_idxs_subset": inlier_indices,
        "X_inlier_subset": np.asarray(X)[inlier_indices],
        "y_inlier_subset": np.asarray(y)[inlier_indices],
    }


@register_atom(witness_ransac_score_callback_payload)
@icontract.require(lambda X_inlier_subset: X_inlier_subset is not None, "X_inlier_subset must be provided")
@icontract.require(lambda y_inlier_subset: y_inlier_subset is not None, "y_inlier_subset must be provided")
@icontract.require(lambda score_params_inlier_subset: _mapping_valid(score_params_inlier_subset), "score_params_inlier_subset must be a mapping")
@icontract.ensure(
    lambda result, X_inlier_subset, y_inlier_subset, score_params_inlier_subset: _score_payload_valid(
        result,
        X_inlier_subset,
        y_inlier_subset,
        score_params_inlier_subset,
    ),
    "score callback payload must preserve inlier subset and sliced score params",
)
def ransac_score_callback_payload(
    X_inlier_subset: object,
    y_inlier_subset: object,
    score_params_inlier_subset: Mapping[str, object],
) -> dict[str, object]:
    """Return the estimator.score callback payload for a RANSAC consensus."""
    return {
        "X_inlier_subset": X_inlier_subset,
        "y_inlier_subset": y_inlier_subset,
        "score_params_inlier_subset": score_params_inlier_subset,
    }


@register_atom(witness_ransac_no_consensus_failure_message)
@icontract.require(lambda skip_limit_exceeded: isinstance(skip_limit_exceeded, bool), "skip_limit_exceeded must be boolean")
@icontract.ensure(
    lambda result, skip_limit_exceeded: result
    == (_MAX_SKIPS_NO_VALID_CONSENSUS_MESSAGE if skip_limit_exceeded else _ALL_TRIALS_SKIPPED_MESSAGE),
    "no-consensus failure message must match RANSAC source wording",
)
def ransac_no_consensus_failure_message(skip_limit_exceeded: bool) -> str:
    """Return RANSAC's no-consensus failure message."""
    if skip_limit_exceeded:
        return _MAX_SKIPS_NO_VALID_CONSENSUS_MESSAGE
    return _ALL_TRIALS_SKIPPED_MESSAGE


@register_atom(witness_ransac_final_refit_callback_payload)
@icontract.require(lambda X_inlier_best: X_inlier_best is not None, "X_inlier_best must be provided")
@icontract.require(lambda y_inlier_best: y_inlier_best is not None, "y_inlier_best must be provided")
@icontract.require(lambda fit_params_best_idxs_subset: _mapping_valid(fit_params_best_idxs_subset), "fit_params_best_idxs_subset must be a mapping")
@icontract.ensure(
    lambda result, X_inlier_best, y_inlier_best, fit_params_best_idxs_subset: _final_refit_payload_valid(
        result,
        X_inlier_best,
        y_inlier_best,
        fit_params_best_idxs_subset,
    ),
    "final refit callback payload must preserve best inliers and sliced fit params",
)
def ransac_final_refit_callback_payload(
    X_inlier_best: object,
    y_inlier_best: object,
    fit_params_best_idxs_subset: Mapping[str, object],
) -> dict[str, object]:
    """Return the final estimator.fit callback payload on best inliers."""
    return {
        "X_inlier_best": X_inlier_best,
        "y_inlier_best": y_inlier_best,
        "fit_params_best_idxs_subset": fit_params_best_idxs_subset,
    }
