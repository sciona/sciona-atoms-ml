"""Permutation feature-score aggregation shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_permutation_importance_feature_scores_are_multimetric,
    witness_permutation_importance_single_feature_score_vector,
)


def _finite_score(value: object) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _score_or_metric_mapping_valid(value: object) -> bool:
    if _finite_score(value):
        return True
    return bool(
        isinstance(value, Mapping)
        and len(value) >= 1
        and all(isinstance(name, str) and len(name) >= 1 and _finite_score(score) for name, score in value.items())
    )


def _single_feature_scores_valid(scores: object) -> bool:
    try:
        array = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _score_vector_valid(result: object, scores: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(scores, dtype=np.float64)
    return bool(values.shape == source.shape and np.all(np.isfinite(values)))


@register_atom(witness_permutation_importance_feature_scores_are_multimetric)
@icontract.require(
    lambda first_score: _score_or_metric_mapping_valid(first_score),
    "first_score must be a finite scalar or a nonempty metric-to-score mapping",
)
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def permutation_importance_feature_scores_are_multimetric(
    first_score: float | Mapping[str, float],
) -> bool:
    """Decide whether one feature's repeated permutation scores are multimetric."""
    return isinstance(first_score, Mapping)


@register_atom(witness_permutation_importance_single_feature_score_vector)
@icontract.require(
    lambda scores: _single_feature_scores_valid(scores),
    "scores must be a finite nonempty one-dimensional score sequence",
)
@icontract.ensure(
    lambda result, scores: _score_vector_valid(result, scores),
    "result must preserve sklearn's one-feature repeated score vector",
)
def permutation_importance_single_feature_score_vector(
    scores: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Coerce one feature's repeated single-metric scores into sklearn's ndarray output."""
    return np.asarray(scores, dtype=np.float64)
