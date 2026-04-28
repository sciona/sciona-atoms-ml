"""SelectFromModel bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_select_from_model_candidate_indices,
    witness_select_from_model_checked_max_features,
    witness_select_from_model_prefit_callable_max_features_ready,
    witness_select_from_model_prefit_estimator_valid,
)


def _feature_count_valid(n_features: int) -> bool:
    return bool(isinstance(n_features, int) and not isinstance(n_features, bool) and n_features >= 1)


def _max_features_valid(max_features: int, n_features: int) -> bool:
    return bool(
        isinstance(max_features, int)
        and not isinstance(max_features, bool)
        and _feature_count_valid(n_features)
        and 0 <= max_features <= n_features
    )


def _finite_scores(scores: NDArray[np.float64]) -> bool:
    values = np.asarray(scores, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)))


def _candidate_indices_valid(result: NDArray[np.int64], scores: NDArray[np.float64], max_features: int) -> bool:
    values = np.asarray(result)
    score_values = np.asarray(scores, dtype=np.float64)
    expected = np.argsort(-score_values, kind="mergesort")[:max_features]
    return bool(
        values.ndim == 1
        and values.shape == (max_features,)
        and np.issubdtype(values.dtype, np.integer)
        and np.array_equal(values, expected.astype(np.int64))
    )


@register_atom(witness_select_from_model_checked_max_features)
@icontract.require(lambda n_features: _feature_count_valid(n_features), "n_features must be a positive integer")
@icontract.require(lambda max_features, n_features: _max_features_valid(max_features, n_features), "max_features must be an integer in [0, n_features]")
@icontract.ensure(lambda result, max_features: isinstance(result, int) and result == max_features, "result must equal the validated max_features value")
def select_from_model_checked_max_features(
    max_features: int,
    *,
    n_features: int,
) -> int:
    """Return sklearn's validated integer `max_features` value after scalar checks."""
    return int(max_features)


@register_atom(witness_select_from_model_prefit_estimator_valid)
@icontract.require(lambda prefit: isinstance(prefit, bool), "prefit must be boolean")
@icontract.require(lambda estimator_is_fitted: isinstance(estimator_is_fitted, bool), "estimator_is_fitted must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def select_from_model_prefit_estimator_valid(
    *,
    prefit: bool,
    estimator_is_fitted: bool,
) -> bool:
    """Return whether sklearn's prefit branch accepts the supplied estimator state."""
    return bool((not prefit) or estimator_is_fitted)


@register_atom(witness_select_from_model_prefit_callable_max_features_ready)
@icontract.require(lambda prefit: isinstance(prefit, bool), "prefit must be boolean")
@icontract.require(lambda max_features_is_callable: isinstance(max_features_is_callable, bool), "max_features_is_callable must be boolean")
@icontract.require(lambda has_fitted_max_features: isinstance(has_fitted_max_features, bool), "has_fitted_max_features must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def select_from_model_prefit_callable_max_features_ready(
    *,
    prefit: bool,
    max_features_is_callable: bool,
    has_fitted_max_features: bool,
) -> bool:
    """Return whether transform-time support-mask logic can use max_features without refitting."""
    return bool(
        not (prefit and max_features_is_callable and not has_fitted_max_features)
    )


@register_atom(witness_select_from_model_candidate_indices)
@icontract.require(lambda scores: _finite_scores(scores), "scores must be a nonempty finite vector")
@icontract.require(lambda scores, max_features: _max_features_valid(max_features, int(np.asarray(scores).shape[0])), "max_features must be in [0, len(scores)]")
@icontract.ensure(lambda result, scores, max_features: _candidate_indices_valid(result, scores, max_features), "candidate indices must match sklearn's stable descending top-k order")
def select_from_model_candidate_indices(
    scores: NDArray[np.float64],
    *,
    max_features: int,
) -> NDArray[np.int64]:
    """Return sklearn's stable descending top-k feature indices for SelectFromModel."""
    values = np.asarray(scores, dtype=np.float64)
    return np.asarray(np.argsort(-values, kind="mergesort")[:max_features], dtype=np.int64)
