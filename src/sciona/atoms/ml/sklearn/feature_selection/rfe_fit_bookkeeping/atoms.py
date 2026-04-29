"""RFE fit-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_rfe_active_feature_indices,
    witness_rfe_resolve_n_features_to_select,
    witness_rfe_resolve_step,
    witness_rfe_step_history_append,
    witness_rfe_warn_too_many_features_to_select,
)

FeatureCountSpec = int | float | None
StepSpec = int | float


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _feature_count_spec_valid(value: FeatureCountSpec) -> bool:
    if value is None:
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return value >= 1
    return bool(isinstance(value, float) and np.isfinite(value) and 0.0 < value <= 1.0)


def _step_spec_valid(value: StepSpec) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return value >= 1
    return bool(isinstance(value, float) and np.isfinite(value) and 0.0 < value < 1.0)


def _support_mask_valid(values: object) -> bool:
    mask = np.asarray(values)
    return bool(mask.ndim == 1 and mask.shape[0] >= 1 and mask.dtype == np.bool_ and np.any(mask))


def _history_inputs_valid(step_n_features: object, step_scores: object) -> bool:
    feature_counts = np.asarray(step_n_features)
    scores = np.asarray(step_scores, dtype=np.float64)
    return bool(
        feature_counts.ndim == 1
        and scores.ndim == 1
        and feature_counts.shape == scores.shape
        and np.issubdtype(feature_counts.dtype, np.integer)
        and np.all(feature_counts >= 1)
        and np.all(np.isfinite(scores))
    )


def _resolved_feature_count_valid(result: int, n_features: int) -> bool:
    return bool(isinstance(result, int) and result >= 1 and result == int(result))


def _resolved_step_valid(result: int) -> bool:
    return bool(isinstance(result, int) and result >= 1)


def _active_feature_indices_valid(result: object, support_mask: object) -> bool:
    values = np.asarray(result)
    mask = np.asarray(support_mask, dtype=np.bool_)
    return bool(
        values.ndim == 1
        and np.issubdtype(values.dtype, np.integer)
        and np.array_equal(values, np.arange(mask.shape[0], dtype=np.int64)[mask])
    )


def _history_append_valid(
    result: tuple[NDArray[np.int64], NDArray[np.float64]],
    step_n_features: object,
    step_scores: object,
) -> bool:
    feature_counts, scores = result
    previous_counts = np.asarray(step_n_features)
    previous_scores = np.asarray(step_scores, dtype=np.float64)
    return bool(
        feature_counts.ndim == 1
        and scores.ndim == 1
        and np.issubdtype(feature_counts.dtype, np.integer)
        and feature_counts.shape[0] == previous_counts.shape[0] + 1
        and scores.shape[0] == previous_scores.shape[0] + 1
        and np.array_equal(feature_counts[:-1], previous_counts.astype(np.int64))
        and np.allclose(scores[:-1], previous_scores)
    )


@register_atom(witness_rfe_resolve_n_features_to_select)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(lambda n_features_to_select: _feature_count_spec_valid(n_features_to_select), "n_features_to_select must be None, a positive integer, or a float in (0, 1]")
@icontract.ensure(lambda result, n_features: _resolved_feature_count_valid(result, n_features), "resolved feature count must be a positive integer")
def rfe_resolve_n_features_to_select(
    n_features: int,
    *,
    n_features_to_select: FeatureCountSpec = None,
) -> int:
    """Resolve sklearn's effective RFE target feature count from the user specification."""
    if n_features_to_select is None:
        return int(n_features // 2)
    if isinstance(n_features_to_select, int) and not isinstance(n_features_to_select, bool):
        return int(n_features_to_select)
    return int(int(n_features) * float(n_features_to_select))


@register_atom(witness_rfe_warn_too_many_features_to_select)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(lambda resolved_n_features_to_select: _positive_int(resolved_n_features_to_select), "resolved_n_features_to_select must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, bool), "warning predicate must be boolean")
def rfe_warn_too_many_features_to_select(
    n_features: int,
    *,
    resolved_n_features_to_select: int,
) -> bool:
    """Return whether sklearn emits the oversize n_features_to_select warning branch."""
    return bool(resolved_n_features_to_select > n_features)


@register_atom(witness_rfe_resolve_step)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(lambda step: _step_spec_valid(step), "step must be a positive integer or a float in (0, 1)")
@icontract.ensure(lambda result: _resolved_step_valid(result), "resolved step must be a positive integer")
def rfe_resolve_step(
    n_features: int,
    *,
    step: StepSpec = 1,
) -> int:
    """Resolve sklearn's effective RFE elimination step from the user specification."""
    if isinstance(step, float):
        return int(max(1, float(step) * int(n_features)))
    return int(step)


@register_atom(witness_rfe_active_feature_indices)
@icontract.require(lambda support_mask: _support_mask_valid(support_mask), "support_mask must be a nonempty boolean vector with at least one active feature")
@icontract.ensure(lambda result, support_mask: _active_feature_indices_valid(result, support_mask), "active feature indices must follow sklearn's support-mask indexing order")
def rfe_active_feature_indices(
    support_mask: NDArray[np.bool_],
) -> NDArray[np.int64]:
    """Return the active feature indices sklearn derives from the current RFE support mask."""
    mask = np.asarray(support_mask, dtype=np.bool_)
    return np.asarray(np.arange(mask.shape[0], dtype=np.int64)[mask], dtype=np.int64)


@register_atom(witness_rfe_step_history_append)
@icontract.require(lambda step_n_features, step_scores: _history_inputs_valid(step_n_features, step_scores), "step_n_features and step_scores must be aligned finite history vectors")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(lambda score: isinstance(score, (int, float)) and not isinstance(score, bool) and np.isfinite(float(score)), "score must be finite")
@icontract.ensure(lambda result, step_n_features, step_scores: _history_append_valid(result, step_n_features, step_scores), "history append must preserve prior entries and add one aligned final entry")
def rfe_step_history_append(
    step_n_features: NDArray[np.int64],
    step_scores: NDArray[np.float64],
    *,
    n_features: int,
    score: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Append one sklearn RFE step-history entry for the current feature count and score."""
    counts = np.asarray(step_n_features, dtype=np.int64)
    scores = np.asarray(step_scores, dtype=np.float64)
    return (
        np.asarray(np.append(counts, int(n_features)), dtype=np.int64),
        np.asarray(np.append(scores, float(score)), dtype=np.float64),
    )
