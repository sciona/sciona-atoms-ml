"""RFE state-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_rfe_elimination_threshold,
    witness_rfe_final_feature_count,
    witness_rfe_initial_ranking,
    witness_rfe_initial_step_history,
    witness_rfe_initial_support_mask,
)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _support_mask_valid(values: object) -> bool:
    mask = np.asarray(values)
    return bool(mask.ndim == 1 and mask.shape[0] >= 1 and mask.dtype == np.bool_)


def _support_mask_all_true(result: object, n_features: int) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (n_features,) and values.dtype == np.bool_ and np.all(values))


def _ranking_valid(result: object, n_features: int) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (n_features,)
        and np.issubdtype(values.dtype, np.integer)
        and np.array_equal(values, np.ones(n_features, dtype=np.int64))
    )


def _history_valid(result: tuple[NDArray[np.int64], NDArray[np.float64]]) -> bool:
    step_n_features, step_scores = result
    return bool(
        step_n_features.ndim == 1
        and step_scores.ndim == 1
        and step_n_features.shape == (0,)
        and step_scores.shape == (0,)
        and np.issubdtype(step_n_features.dtype, np.integer)
    )


def _threshold_valid(result: int, active_feature_count: int, n_features_to_select: int) -> bool:
    return bool(
        isinstance(result, int)
        and result >= 1
        and result <= active_feature_count - n_features_to_select
    )


def _final_feature_count_valid(result: int, support_mask: object) -> bool:
    mask = np.asarray(support_mask, dtype=np.bool_)
    return bool(isinstance(result, int) and result == int(np.sum(mask)) and result >= 0)


@register_atom(witness_rfe_initial_support_mask)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.ensure(lambda result, n_features: _support_mask_all_true(result, n_features), "initial support mask must be an all-True boolean vector")
def rfe_initial_support_mask(
    n_features: int,
) -> NDArray[np.bool_]:
    """Initialize sklearn's all-active support mask for the RFE fit loop."""
    return np.ones(int(n_features), dtype=np.bool_)


@register_atom(witness_rfe_initial_ranking)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.ensure(lambda result, n_features: _ranking_valid(result, n_features), "initial ranking must be an all-ones integer vector")
def rfe_initial_ranking(
    n_features: int,
) -> NDArray[np.int64]:
    """Initialize sklearn's all-ones ranking vector for the RFE fit loop."""
    return np.ones(int(n_features), dtype=np.int64)


@register_atom(witness_rfe_initial_step_history)
@icontract.require(lambda: True, "RFE step-history initialization has no input preconditions")
@icontract.ensure(lambda result: _history_valid(result), "initial step history must contain empty feature-count and score vectors")
def rfe_initial_step_history() -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Initialize sklearn's optional RFE step history as empty arrays."""
    return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)


@register_atom(witness_rfe_elimination_threshold)
@icontract.require(lambda active_feature_count: _positive_int(active_feature_count), "active_feature_count must be a positive integer")
@icontract.require(lambda n_features_to_select: _positive_int(n_features_to_select), "n_features_to_select must be a positive integer")
@icontract.require(lambda step: _positive_int(step), "step must be a positive integer")
@icontract.require(lambda active_feature_count, n_features_to_select: active_feature_count > n_features_to_select, "active_feature_count must exceed n_features_to_select")
@icontract.ensure(lambda result, active_feature_count, n_features_to_select: _threshold_valid(result, active_feature_count, n_features_to_select), "threshold must be sklearn's bounded elimination count")
def rfe_elimination_threshold(
    active_feature_count: int,
    *,
    n_features_to_select: int,
    step: int,
) -> int:
    """Resolve sklearn's bounded feature-elimination count for one RFE iteration."""
    return int(min(step, active_feature_count - n_features_to_select))


@register_atom(witness_rfe_final_feature_count)
@icontract.require(lambda support_mask: _support_mask_valid(support_mask), "support_mask must be a nonempty boolean vector")
@icontract.ensure(lambda result, support_mask: _final_feature_count_valid(result, support_mask), "final feature count must equal the number of active support entries")
def rfe_final_feature_count(
    support_mask: NDArray[np.bool_],
) -> int:
    """Count sklearn's final selected features from the terminal RFE support mask."""
    return int(np.sum(np.asarray(support_mask, dtype=np.bool_)))
