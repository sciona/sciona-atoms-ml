"""RANSAC consensus bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_ransac_consensus_is_better,
    witness_ransac_default_residual_threshold,
    witness_ransac_dynamic_max_trials,
    witness_ransac_inlier_mask,
    witness_ransac_loss_residuals,
)

_EPSILON = np.spacing(1)


def _finite_target(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim in {1, 2} and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _aligned_targets(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> bool:
    return bool(_finite_target(y_true) and _finite_target(y_pred) and np.asarray(y_true).shape == np.asarray(y_pred).shape)


def _finite_vector(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _loss_valid(loss: str) -> bool:
    return loss in {"absolute_error", "squared_error"}


def _threshold_valid(residual_threshold: float) -> bool:
    return bool(
        isinstance(residual_threshold, (int, float))
        and not isinstance(residual_threshold, bool)
        and np.isfinite(float(residual_threshold))
        and float(residual_threshold) >= 0.0
    )


def _count_valid(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _score_valid(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _trial_inputs_valid(n_inliers: int, n_samples: int, min_samples: int, probability: float) -> bool:
    return bool(
        isinstance(n_inliers, int)
        and not isinstance(n_inliers, bool)
        and isinstance(n_samples, int)
        and not isinstance(n_samples, bool)
        and isinstance(min_samples, int)
        and not isinstance(min_samples, bool)
        and 0 <= n_inliers <= n_samples
        and 1 <= min_samples <= n_samples
        and isinstance(probability, (int, float))
        and not isinstance(probability, bool)
        and np.isfinite(float(probability))
        and 0.0 <= float(probability) <= 1.0
    )


def _scalar_result_valid(result: float) -> bool:
    return bool(isinstance(result, float) and np.isfinite(result))


def _residual_result_valid(result: NDArray[np.float64], y_true: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    target = np.asarray(y_true)
    return bool(values.shape == (target.shape[0],) and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _mask_result_valid(result: NDArray[np.bool_], residuals: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    residual_values = np.asarray(residuals)
    return bool(values.dtype == np.bool_ and values.shape == residual_values.shape)


def _trial_result_valid(result: float) -> bool:
    return bool(isinstance(result, float) and (np.isfinite(result) or np.isinf(result)) and result >= 0.0)


@register_atom(witness_ransac_default_residual_threshold)
@icontract.require(lambda y: _finite_target(y), "y must be a finite 1D or 2D target array")
@icontract.ensure(lambda result: _scalar_result_valid(result), "default residual threshold must be finite")
def ransac_default_residual_threshold(y: NDArray[np.float64]) -> float:
    """Compute the default residual threshold from target spread."""
    target = np.asarray(y, dtype=np.float64)
    return float(np.median(np.abs(target - np.median(target))))


@register_atom(witness_ransac_loss_residuals)
@icontract.require(lambda y_true, y_pred: _aligned_targets(y_true, y_pred), "targets and predictions must be finite and aligned")
@icontract.require(lambda loss: _loss_valid(loss), "loss must be absolute_error or squared_error")
@icontract.ensure(lambda result, y_true: _residual_result_valid(result, y_true), "residuals must contain one nonnegative value per sample")
def ransac_loss_residuals(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    *,
    loss: str = "absolute_error",
) -> NDArray[np.float64]:
    """Compute per-sample residual values for built-in loss choices."""
    true_values = np.asarray(y_true, dtype=np.float64)
    pred_values = np.asarray(y_pred, dtype=np.float64)
    if loss == "absolute_error":
        residuals = np.abs(true_values - pred_values)
    else:
        residuals = (true_values - pred_values) ** 2
    if residuals.ndim == 2:
        residuals = np.sum(residuals, axis=1)
    return np.asarray(residuals, dtype=np.float64)


@register_atom(witness_ransac_inlier_mask)
@icontract.require(lambda residuals: _finite_vector(residuals), "residuals must be a finite 1D vector")
@icontract.require(lambda residual_threshold: _threshold_valid(residual_threshold), "residual_threshold must be finite and nonnegative")
@icontract.ensure(lambda result, residuals: _mask_result_valid(result, residuals), "inlier mask must align with residuals")
def ransac_inlier_mask(
    residuals: NDArray[np.float64],
    *,
    residual_threshold: float,
) -> NDArray[np.bool_]:
    """Classify samples as inliers when residuals are within the threshold."""
    return np.asarray(np.asarray(residuals, dtype=np.float64) <= float(residual_threshold), dtype=np.bool_)


@register_atom(witness_ransac_consensus_is_better)
@icontract.require(lambda n_inliers: _count_valid(n_inliers), "n_inliers must be nonnegative")
@icontract.require(lambda best_n_inliers: _count_valid(best_n_inliers), "best_n_inliers must be nonnegative")
@icontract.require(lambda score: _score_valid(score), "score must be finite")
@icontract.require(lambda best_score: _score_valid(best_score), "best_score must be finite")
@icontract.ensure(lambda result: isinstance(result, bool), "consensus decision must be boolean")
def ransac_consensus_is_better(
    n_inliers: int,
    score: float,
    best_n_inliers: int,
    best_score: float,
) -> bool:
    """Decide whether a candidate consensus should replace the current best."""
    if int(n_inliers) < int(best_n_inliers):
        return False
    if int(n_inliers) == int(best_n_inliers) and float(score) < float(best_score):
        return False
    return True


@register_atom(witness_ransac_dynamic_max_trials)
@icontract.require(lambda n_inliers, n_samples, min_samples, probability: _trial_inputs_valid(n_inliers, n_samples, min_samples, probability), "trial inputs must describe a valid consensus state")
@icontract.ensure(lambda result: _trial_result_valid(result), "trial bound must be nonnegative or infinite")
def ransac_dynamic_max_trials(
    n_inliers: int,
    n_samples: int,
    min_samples: int,
    probability: float,
) -> float:
    """Compute the trial bound needed to sample one all-inlier subset."""
    inlier_ratio = int(n_inliers) / float(n_samples)
    nom = max(_EPSILON, 1.0 - float(probability))
    denom = max(_EPSILON, 1.0 - inlier_ratio**int(min_samples))
    if nom == 1.0:
        return 0.0
    if denom == 1.0:
        return float("inf")
    return abs(float(np.ceil(np.log(nom) / np.log(denom))))
