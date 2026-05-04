"""Permutation feature multimetric aggregation atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_permutation_importance_feature_metric_names,
    witness_permutation_importance_feature_metric_score_dict,
)

ScoreDict = Mapping[str, float]


def _finite_score(value: object) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _score_dicts_valid(score_dicts: object) -> bool:
    if not isinstance(score_dicts, tuple) or len(score_dicts) < 1:
        return False
    first = score_dicts[0]
    if not isinstance(first, Mapping) or len(first) < 1:
        return False
    expected_keys = tuple(first.keys())
    return bool(
        all(isinstance(name, str) and len(name) >= 1 for name in expected_keys)
        and all(
            isinstance(score_dict, Mapping)
            and tuple(score_dict.keys()) == expected_keys
            and all(_finite_score(score_dict[name]) for name in expected_keys)
            for score_dict in score_dicts
        )
    )


def _metric_names_valid(result: object, score_dicts: tuple[ScoreDict, ...]) -> bool:
    return bool(
        isinstance(result, tuple)
        and len(result) == len(score_dicts[0])
        and result == tuple(score_dicts[0].keys())
        and all(isinstance(name, str) and len(name) >= 1 for name in result)
    )


def _metric_score_dict_valid(result: object, score_dicts: tuple[ScoreDict, ...]) -> bool:
    if not isinstance(result, dict):
        return False
    expected_keys = tuple(score_dicts[0].keys())
    if tuple(result.keys()) != expected_keys:
        return False
    n_repeats = len(score_dicts)
    for key in expected_keys:
        values = np.asarray(result[key], dtype=np.float64)
        if values.shape != (n_repeats,) or not np.all(np.isfinite(values)):
            return False
    return True


@register_atom(witness_permutation_importance_feature_metric_names)
@icontract.require(
    lambda score_dicts: _score_dicts_valid(score_dicts),
    "score_dicts must be a nonempty tuple of same-key finite score dictionaries",
)
@icontract.ensure(
    lambda result, score_dicts: _metric_names_valid(result, score_dicts),
    "metric names must preserve sklearn's first-dict key iteration order",
)
def permutation_importance_feature_metric_names(
    score_dicts: tuple[ScoreDict, ...],
) -> tuple[str, ...]:
    """Resolve the metric iteration order used by _aggregate_score_dicts for one feature."""
    return tuple(score_dicts[0].keys())


@register_atom(witness_permutation_importance_feature_metric_score_dict)
@icontract.require(
    lambda score_dicts: _score_dicts_valid(score_dicts),
    "score_dicts must be a nonempty tuple of same-key finite score dictionaries",
)
@icontract.ensure(
    lambda result, score_dicts: _metric_score_dict_valid(result, score_dicts),
    "result must match sklearn's numeric dict-of-score-vectors aggregation",
)
def permutation_importance_feature_metric_score_dict(
    score_dicts: tuple[ScoreDict, ...],
) -> dict[str, NDArray[np.float64]]:
    """Aggregate one feature's repeated numeric score dicts into sklearn's dict-of-vectors form."""
    return {
        key: np.asarray([score_dict[key] for score_dict in score_dicts], dtype=np.float64)
        for key in score_dicts[0]
    }
