"""SequentialFeatureSelector fit-shell bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    SequentialFeatureCountSpec,
    witness_sequential_auto_select_enabled,
    witness_sequential_direction_tol_valid,
    witness_sequential_finalize_support,
    witness_sequential_iteration_count,
    witness_sequential_resolve_n_features_to_select,
    witness_sequential_tolerance_break,
)


def _positive_feature_count(n_features: int) -> bool:
    return bool(isinstance(n_features, int) and not isinstance(n_features, bool) and n_features >= 2)


def _selection_spec_valid(n_features_to_select: SequentialFeatureCountSpec) -> bool:
    if n_features_to_select == "auto":
        return True
    if isinstance(n_features_to_select, int) and not isinstance(n_features_to_select, bool):
        return n_features_to_select >= 1
    if isinstance(n_features_to_select, float):
        return bool(np.isfinite(n_features_to_select) and 0.0 < n_features_to_select <= 1.0)
    return False


def _tol_valid(tol: float | None) -> bool:
    return tol is None or (isinstance(tol, (int, float)) and not isinstance(tol, bool) and np.isfinite(float(tol)))


def _direction_valid(direction: str) -> bool:
    return direction in {"forward", "backward"}


def _resolved_count_valid(result: int, n_features: int) -> bool:
    return bool(isinstance(result, int) and not isinstance(result, bool) and 1 <= result < n_features)


def _mask_valid(current_mask: NDArray[np.bool_]) -> bool:
    values = np.asarray(current_mask)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and values.dtype == np.bool_)


def _support_result_valid(result: NDArray[np.bool_], current_mask: NDArray[np.bool_]) -> bool:
    values = np.asarray(result)
    current_values = np.asarray(current_mask)
    return bool(values.shape == current_values.shape and values.dtype == np.bool_)


def _finite_score(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _score_or_neg_inf(value: float) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (np.isfinite(float(value)) or float(value) == -np.inf)
    )


@register_atom(witness_sequential_resolve_n_features_to_select)
@icontract.require(lambda n_features: _positive_feature_count(n_features), "n_features must be an integer at least 2")
@icontract.require(lambda n_features_to_select: _selection_spec_valid(n_features_to_select), "n_features_to_select must be 'auto', a positive int, or a float in (0, 1]")
@icontract.require(lambda tol: _tol_valid(tol), "tol must be finite or None")
@icontract.require(
    lambda n_features, n_features_to_select: not (
        isinstance(n_features_to_select, int)
        and not isinstance(n_features_to_select, bool)
        and n_features_to_select >= n_features
    ),
    "integer n_features_to_select must be smaller than n_features",
)
@icontract.ensure(lambda result, n_features: _resolved_count_valid(result, n_features), "resolved feature count must lie in [1, n_features)")
def sequential_resolve_n_features_to_select(
    n_features: int,
    *,
    n_features_to_select: SequentialFeatureCountSpec = "auto",
    tol: float | None = None,
) -> int:
    """Resolve sklearn's fitted `n_features_to_select_` value before the main loop."""
    if n_features_to_select == "auto":
        if tol is not None:
            return int(n_features - 1)
        return int(n_features // 2)
    if isinstance(n_features_to_select, int) and not isinstance(n_features_to_select, bool):
        return int(n_features_to_select)
    return int(n_features * float(n_features_to_select))


@register_atom(witness_sequential_direction_tol_valid)
@icontract.require(lambda direction: _direction_valid(direction), "direction must be forward or backward")
@icontract.require(lambda tol: _tol_valid(tol), "tol must be finite or None")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def sequential_direction_tol_valid(
    *,
    direction: str = "forward",
    tol: float | None = None,
) -> bool:
    """Return whether sklearn's direction-specific tolerance guard allows fit to proceed."""
    return bool(not (tol is not None and float(tol) < 0.0 and direction == "forward"))


@register_atom(witness_sequential_auto_select_enabled)
@icontract.require(lambda n_features_to_select: _selection_spec_valid(n_features_to_select), "n_features_to_select must be 'auto', a positive int, or a float in (0, 1]")
@icontract.require(lambda tol: _tol_valid(tol), "tol must be finite or None")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def sequential_auto_select_enabled(
    *,
    n_features_to_select: SequentialFeatureCountSpec = "auto",
    tol: float | None = None,
) -> bool:
    """Return whether sklearn will use tolerance-based early stopping in the main loop."""
    return bool(tol is not None and n_features_to_select == "auto")


@register_atom(witness_sequential_iteration_count)
@icontract.require(lambda n_features: _positive_feature_count(n_features), "n_features must be an integer at least 2")
@icontract.require(lambda n_features_to_select_resolved, n_features: _resolved_count_valid(n_features_to_select_resolved, n_features), "resolved feature count must lie in [1, n_features)")
@icontract.require(lambda n_features_to_select: _selection_spec_valid(n_features_to_select), "n_features_to_select must be 'auto', a positive int, or a float in (0, 1]")
@icontract.require(lambda direction: _direction_valid(direction), "direction must be forward or backward")
@icontract.ensure(lambda result, n_features: isinstance(result, int) and 1 <= result <= n_features - 1, "iteration count must be a positive integer below n_features")
def sequential_iteration_count(
    n_features: int,
    n_features_to_select_resolved: int,
    *,
    n_features_to_select: SequentialFeatureCountSpec = "auto",
    direction: str = "forward",
) -> int:
    """Compute sklearn's main-loop iteration count for sequential feature selection."""
    if n_features_to_select == "auto" or direction == "forward":
        return int(n_features_to_select_resolved)
    return int(n_features - n_features_to_select_resolved)


@register_atom(witness_sequential_tolerance_break)
@icontract.require(lambda old_score: _score_or_neg_inf(old_score), "old_score must be finite or negative infinity")
@icontract.require(lambda new_score: _finite_score(new_score), "new_score must be finite")
@icontract.require(lambda tol: _finite_score(tol), "tol must be finite")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def sequential_tolerance_break(
    old_score: float,
    new_score: float,
    *,
    tol: float,
) -> bool:
    """Return whether sklearn's auto-selection loop would stop at the current score improvement."""
    return bool((float(new_score) - float(old_score)) < float(tol))


@register_atom(witness_sequential_finalize_support)
@icontract.require(lambda current_mask: _mask_valid(current_mask), "current_mask must be a nonempty boolean vector")
@icontract.require(lambda direction: _direction_valid(direction), "direction must be forward or backward")
@icontract.ensure(lambda result, current_mask: _support_result_valid(result, current_mask), "support mask must preserve the current-mask shape")
def sequential_finalize_support(
    current_mask: NDArray[np.bool_],
    *,
    direction: str = "forward",
) -> NDArray[np.bool_]:
    """Finalize sklearn's selected-feature mask after the main sequential loop."""
    current = np.asarray(current_mask, dtype=np.bool_)
    if direction == "backward":
        return np.asarray(~current, dtype=np.bool_)
    return np.asarray(current, dtype=np.bool_)
