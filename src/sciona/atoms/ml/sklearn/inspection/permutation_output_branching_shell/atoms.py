"""Permutation output-branching shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_permutation_importance_metric_names,
    witness_permutation_importance_single_metric_score_matrix,
    witness_permutation_importance_use_multimetric_results,
)


def _finite_score(value: object) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _baseline_score_or_dict_valid(value: object) -> bool:
    if _finite_score(value):
        return True
    return bool(
        isinstance(value, Mapping)
        and len(value) >= 1
        and all(isinstance(name, str) and len(name) >= 1 and _finite_score(score) for name, score in value.items())
    )


def _metric_name_mapping_valid(value: object) -> bool:
    return bool(
        isinstance(value, Mapping)
        and len(value) >= 1
        and all(isinstance(name, str) and len(name) >= 1 and _finite_score(score) for name, score in value.items())
    )


def _metric_names_valid(result: object, baseline_scores: Mapping[str, float]) -> bool:
    return bool(
        isinstance(result, tuple)
        and len(result) == len(baseline_scores)
        and result == tuple(baseline_scores.keys())
        and all(isinstance(name, str) and len(name) >= 1 for name in result)
    )


def _single_metric_scores_valid(scores: object) -> bool:
    try:
        array = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _score_matrix_valid(result: object, scores: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(scores, dtype=np.float64)
    return bool(values.shape == source.shape and np.all(np.isfinite(values)))


@register_atom(witness_permutation_importance_use_multimetric_results)
@icontract.require(
    lambda baseline_score: _baseline_score_or_dict_valid(baseline_score),
    "baseline_score must be a finite scalar or a nonempty metric-to-score mapping",
)
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def permutation_importance_use_multimetric_results(
    baseline_score: float | Mapping[str, float],
) -> bool:
    """Decide whether permutation_importance uses the multimetric return branch."""
    return isinstance(baseline_score, Mapping)


@register_atom(witness_permutation_importance_metric_names)
@icontract.require(
    lambda baseline_scores: _metric_name_mapping_valid(baseline_scores),
    "baseline_scores must be a nonempty metric-to-score mapping",
)
@icontract.ensure(
    lambda result, baseline_scores: _metric_names_valid(result, baseline_scores),
    "metric names must preserve sklearn's baseline-score iteration order",
)
def permutation_importance_metric_names(
    baseline_scores: Mapping[str, float],
) -> tuple[str, ...]:
    """Resolve the metric iteration order used by permutation_importance."""
    return tuple(baseline_scores.keys())


@register_atom(witness_permutation_importance_single_metric_score_matrix)
@icontract.require(
    lambda scores: _single_metric_scores_valid(scores),
    "scores must be a finite nonempty 2D single-metric score matrix",
)
@icontract.ensure(
    lambda result, scores: _score_matrix_valid(result, scores),
    "result must preserve sklearn's single-metric feature-by-repeat score matrix",
)
def permutation_importance_single_metric_score_matrix(
    scores: Sequence[Sequence[float]] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Coerce single-metric per-feature scores into sklearn's final score matrix."""
    return np.asarray(scores, dtype=np.float64)
