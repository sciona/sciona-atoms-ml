"""Estimator-independent selector bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    ThresholdSpec,
    TransformSpec,
    witness_feature_importances_transform,
    witness_rfe_elimination_step,
    witness_select_from_model_support_mask,
    witness_select_from_model_threshold,
    witness_sequential_best_feature,
    witness_sequential_candidate_masks,
)


def _finite_vector(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_1d_or_2d(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.ndim in {1, 2} and array.size >= 1 and np.all(np.isfinite(array)))


def _transform_valid(transform_func: TransformSpec) -> bool:
    return transform_func in {None, "norm", "square"}


def _norm_order_valid(norm_order: int) -> bool:
    return bool(isinstance(norm_order, int) and not isinstance(norm_order, bool) and norm_order >= 1)


def _threshold_spec_valid(threshold: ThresholdSpec) -> bool:
    if threshold is None:
        return True
    if isinstance(threshold, str):
        if threshold in {"mean", "median"}:
            return True
        if "*" not in threshold:
            return False
        scale, reference = threshold.split("*", maxsplit=1)
        try:
            float(scale.strip())
        except ValueError:
            return False
        return reference.strip() in {"mean", "median"}
    return bool(isinstance(threshold, (int, float)) and not isinstance(threshold, bool) and np.isfinite(float(threshold)))


def _max_features_valid(max_features: int | None, n_features: int) -> bool:
    return bool(
        max_features is None
        or (
            isinstance(max_features, int)
            and not isinstance(max_features, bool)
            and 0 <= max_features <= n_features
        )
    )


def _support_mask_valid(mask: NDArray[np.bool_]) -> bool:
    values = np.asarray(mask)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and values.dtype == np.bool_)


def _ranking_valid(ranking: NDArray[np.int64], n_features: int) -> bool:
    values = np.asarray(ranking)
    return bool(values.ndim == 1 and values.shape[0] == n_features and np.issubdtype(values.dtype, np.integer) and np.all(values >= 1))


def _rfe_inputs_valid(
    support_mask: NDArray[np.bool_],
    ranking: NDArray[np.int64],
    importances: NDArray[np.float64],
    n_features_to_select: int,
    step: int,
) -> bool:
    support = np.asarray(support_mask)
    importances_values = np.asarray(importances, dtype=np.float64)
    active_count = int(np.sum(support)) if support.ndim == 1 else 0
    return bool(
        _support_mask_valid(support_mask)
        and _ranking_valid(ranking, support.shape[0])
        and _finite_vector(importances)
        and importances_values.shape[0] == active_count
        and isinstance(n_features_to_select, int)
        and not isinstance(n_features_to_select, bool)
        and 1 <= n_features_to_select < active_count
        and isinstance(step, int)
        and not isinstance(step, bool)
        and step >= 1
    )


def _candidate_mask_valid(current_mask: NDArray[np.bool_]) -> bool:
    values = np.asarray(current_mask)
    return bool(_support_mask_valid(current_mask) and np.any(~values))


def _candidate_scores_valid(candidate_indices: NDArray[np.int64], scores: NDArray[np.float64]) -> bool:
    indices = np.asarray(candidate_indices)
    score_values = np.asarray(scores, dtype=np.float64)
    return bool(
        indices.ndim == 1
        and indices.shape[0] >= 1
        and np.issubdtype(indices.dtype, np.integer)
        and np.all(indices >= 0)
        and score_values.ndim == 1
        and score_values.shape == indices.shape
        and np.all(np.isfinite(score_values))
    )


def _importance_result_valid(result: NDArray[np.float64], importances: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(importances, dtype=np.float64)
    n_features = input_values.shape[0] if input_values.ndim == 1 else input_values.shape[1]
    return bool(values.shape == (n_features,) and np.all(np.isfinite(values)))


def _threshold_result_valid(result: float) -> bool:
    return bool(isinstance(result, float) and np.isfinite(result))


def _mask_result_valid(result: NDArray[np.bool_], scores: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    score_values = np.asarray(scores)
    return bool(values.dtype == np.bool_ and values.shape == score_values.shape)


def _rfe_result_valid(result: tuple[NDArray[np.bool_], NDArray[np.int64]], support_mask: NDArray[np.bool_], ranking: NDArray[np.int64]) -> bool:
    support, ranking_result = result
    return bool(_support_mask_valid(support) and support.shape == np.asarray(support_mask).shape and _ranking_valid(ranking_result, np.asarray(ranking).shape[0]))


def _candidate_result_valid(result: tuple[NDArray[np.int64], NDArray[np.bool_]], current_mask: NDArray[np.bool_]) -> bool:
    indices, masks = result
    index_values = np.asarray(indices)
    mask_values = np.asarray(masks)
    n_features = np.asarray(current_mask).shape[0]
    return bool(
        index_values.ndim == 1
        and np.issubdtype(index_values.dtype, np.integer)
        and mask_values.dtype == np.bool_
        and mask_values.shape == (index_values.shape[0], n_features)
    )


@register_atom(witness_feature_importances_transform)
@icontract.require(lambda importances: _finite_1d_or_2d(importances), "importances must be finite and 1D or 2D")
@icontract.require(lambda transform_func: _transform_valid(transform_func), "transform_func must be None, norm, or square")
@icontract.require(lambda importances, transform_func: transform_func is not None or np.asarray(importances).ndim == 1, "2D importances require an aggregation transform")
@icontract.require(lambda norm_order: _norm_order_valid(norm_order), "norm_order must be a positive integer")
@icontract.ensure(lambda result, importances: _importance_result_valid(result, importances), "transformed importances must produce one finite score per feature")
def feature_importances_transform(
    importances: NDArray[np.float64],
    *,
    transform_func: TransformSpec = None,
    norm_order: int = 1,
) -> NDArray[np.float64]:
    """Transform retrieved estimator importances into one score per feature."""
    values = np.asarray(importances, dtype=np.float64)
    if transform_func is None:
        return np.asarray(np.ravel(values), dtype=np.float64)
    if transform_func == "norm":
        if values.ndim == 1:
            return np.asarray(np.abs(values), dtype=np.float64)
        return np.asarray(np.linalg.norm(values, axis=0, ord=norm_order), dtype=np.float64)
    if values.ndim == 1:
        return np.asarray(values**2, dtype=np.float64)
    return np.asarray((values**2).sum(axis=0), dtype=np.float64)


@register_atom(witness_select_from_model_threshold)
@icontract.require(lambda importances: _finite_vector(importances), "importances must be a finite 1D score vector")
@icontract.require(lambda threshold: _threshold_spec_valid(threshold), "threshold must be numeric, mean, median, scale*mean, scale*median, or None")
@icontract.ensure(lambda result: _threshold_result_valid(result), "threshold must resolve to a finite scalar")
def select_from_model_threshold(
    importances: NDArray[np.float64],
    *,
    threshold: ThresholdSpec = None,
    l1_default: bool = False,
) -> float:
    """Resolve sklearn SelectFromModel threshold text against importance scores."""
    values = np.asarray(importances, dtype=np.float64)
    resolved: str | float
    if threshold is None:
        resolved = 1e-5 if l1_default else "mean"
    else:
        resolved = threshold

    if isinstance(resolved, str):
        if "*" in resolved:
            scale_text, reference_text = resolved.split("*", maxsplit=1)
            scale = float(scale_text.strip())
            reference = reference_text.strip()
            reference_value = np.median(values) if reference == "median" else np.mean(values)
            return float(scale * reference_value)
        if resolved == "median":
            return float(np.median(values))
        return float(np.mean(values))
    return float(resolved)


@register_atom(witness_select_from_model_support_mask)
@icontract.require(lambda scores: _finite_vector(scores), "scores must be a finite 1D score vector")
@icontract.require(lambda scores, max_features: _max_features_valid(max_features, np.asarray(scores).shape[0]), "max_features must be None or a valid count")
@icontract.require(lambda threshold: isinstance(threshold, (int, float)) and not isinstance(threshold, bool) and np.isfinite(float(threshold)), "threshold must be finite")
@icontract.ensure(lambda result, scores: _mask_result_valid(result, scores), "support mask must match score shape")
def select_from_model_support_mask(
    scores: NDArray[np.float64],
    *,
    threshold: float,
    max_features: int | None = None,
) -> NDArray[np.bool_]:
    """Build sklearn SelectFromModel support mask from supplied scores."""
    values = np.asarray(scores, dtype=np.float64)
    if max_features is not None:
        mask = np.zeros_like(values, dtype=np.bool_)
        candidate_indices = np.argsort(-values, kind="mergesort")[:max_features]
        mask[candidate_indices] = True
    else:
        mask = np.ones_like(values, dtype=np.bool_)
    mask[values < float(threshold)] = False
    return np.asarray(mask, dtype=np.bool_)


@register_atom(witness_rfe_elimination_step)
@icontract.require(lambda support_mask, ranking, importances, n_features_to_select, step: _rfe_inputs_valid(support_mask, ranking, importances, n_features_to_select, step), "RFE step inputs must describe an active elimination state")
@icontract.ensure(lambda result, support_mask, ranking: _rfe_result_valid(result, support_mask, ranking), "RFE step must return support and ranking vectors")
def rfe_elimination_step(
    support_mask: NDArray[np.bool_],
    ranking: NDArray[np.int64],
    importances: NDArray[np.float64],
    *,
    n_features_to_select: int,
    step: int,
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """Apply one sklearn RFE elimination update from current importances."""
    support = np.asarray(support_mask, dtype=np.bool_).copy()
    ranking_values = np.asarray(ranking, dtype=np.int64).copy()
    features = np.arange(support.shape[0], dtype=np.int64)[support]
    ranks = np.ravel(np.argsort(np.asarray(importances, dtype=np.float64)))
    threshold = min(int(step), int(np.sum(support)) - int(n_features_to_select))
    support[features[ranks][:threshold]] = False
    ranking_values[np.logical_not(support)] += 1
    return np.asarray(support, dtype=np.bool_), np.asarray(ranking_values, dtype=np.int64)


@register_atom(witness_sequential_candidate_masks)
@icontract.require(lambda current_mask: _candidate_mask_valid(current_mask), "current_mask must be a non-full support mask")
@icontract.require(lambda direction: direction in {"forward", "backward"}, "direction must be forward or backward")
@icontract.ensure(lambda result, current_mask: _candidate_result_valid(result, current_mask), "candidate masks must align with candidate indices")
def sequential_candidate_masks(
    current_mask: NDArray[np.bool_],
    *,
    direction: str = "forward",
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Create sklearn sequential feature-selection candidate masks."""
    current = np.asarray(current_mask, dtype=np.bool_)
    candidate_indices = np.flatnonzero(~current).astype(np.int64)
    masks = []
    for feature_idx in candidate_indices:
        candidate_mask = current.copy()
        candidate_mask[feature_idx] = True
        if direction == "backward":
            candidate_mask = ~candidate_mask
        masks.append(candidate_mask)
    return candidate_indices, np.asarray(masks, dtype=np.bool_)


@register_atom(witness_sequential_best_feature)
@icontract.require(lambda candidate_indices, scores: _candidate_scores_valid(candidate_indices, scores), "candidate_indices and scores must be aligned vectors")
@icontract.ensure(lambda result, candidate_indices: int(result) in set(np.asarray(candidate_indices, dtype=np.int64).tolist()), "selected feature must be one of the candidates")
def sequential_best_feature(
    candidate_indices: NDArray[np.int64],
    scores: NDArray[np.float64],
) -> int:
    """Choose the candidate feature with largest sequential-selection score."""
    indices = np.asarray(candidate_indices, dtype=np.int64)
    score_values = np.asarray(scores, dtype=np.float64)
    return int(indices[int(np.argmax(score_values))])
