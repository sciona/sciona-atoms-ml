"""Permutation-importance result-packaging atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.atoms.ml.sklearn.inspection.permutation import (
    permutation_importance_mean,
    permutation_importance_std,
    permutation_importance_values,
)
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_permutation_importance_metric_score_matrix,
    witness_permutation_importance_multi_metric_bunches,
    witness_permutation_importance_random_seed,
    witness_permutation_importance_summary_bunch,
)

ScoreDictTuple = tuple[dict[str, NDArray[np.float64]], ...]

def _random_state_valid(random_state: object) -> bool:
    return random_state is None or (isinstance(random_state, int) and not isinstance(random_state, bool))

def _int32_seed_valid(result: object) -> bool:
    return bool(
        isinstance(result, int)
        and 0 <= result <= np.iinfo(np.int32).max
    )

def _finite_score(value: object) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))

def _score_dicts_valid(score_dicts_by_feature: object) -> bool:
    if not isinstance(score_dicts_by_feature, tuple) or len(score_dicts_by_feature) < 1:
        return False
    first = score_dicts_by_feature[0]
    if not isinstance(first, dict) or len(first) < 1:
        return False
    expected_keys = tuple(first.keys())
    expected_lengths: dict[str, int] = {}
    for key, values in first.items():
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if array.ndim != 1 or array.shape[0] < 1 or not np.all(np.isfinite(array)):
            return False
        expected_lengths[key] = int(array.shape[0])
    for score_dict in score_dicts_by_feature:
        if not isinstance(score_dict, dict) or tuple(score_dict.keys()) != expected_keys:
            return False
        for key in expected_keys:
            try:
                array = np.asarray(score_dict[key], dtype=np.float64)
            except (TypeError, ValueError):
                return False
            if array.ndim != 1 or array.shape[0] != expected_lengths[key] or not np.all(np.isfinite(array)):
                return False
    return True

def _metric_name_valid(metric_name: object, score_dicts_by_feature: ScoreDictTuple) -> bool:
    return bool(
        isinstance(metric_name, str)
        and len(metric_name) >= 1
        and _score_dicts_valid(score_dicts_by_feature)
        and metric_name in score_dicts_by_feature[0]
    )

def _metric_score_matrix_valid(result: object, score_dicts_by_feature: ScoreDictTuple, metric_name: str) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_features = len(score_dicts_by_feature)
    n_repeats = np.asarray(score_dicts_by_feature[0][metric_name], dtype=np.float64).shape[0]
    return bool(values.shape == (n_features, n_repeats) and np.all(np.isfinite(values)))

def _finite_score_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))

def _summary_bunch_valid(result: object, permuted_scores: object) -> bool:
    from sklearn.utils import Bunch, check_random_state
    if not isinstance(result, Bunch):
        return False
    values = np.asarray(permuted_scores, dtype=np.float64)
    if not {"importances_mean", "importances_std", "importances"}.issubset(result.keys()):
        return False
    return bool(
        np.asarray(result["importances_mean"]).shape == (values.shape[0],)
        and np.asarray(result["importances_std"]).shape == (values.shape[0],)
        and np.asarray(result["importances"]).shape == values.shape
    )

def _baseline_scores_valid(baseline_scores: object) -> bool:
    return bool(
        isinstance(baseline_scores, Mapping)
        and len(baseline_scores) >= 1
        and all(isinstance(name, str) and len(name) >= 1 and _finite_score(score) for name, score in baseline_scores.items())
    )

def _multi_metric_inputs_valid(baseline_scores: object, score_dicts_by_feature: object) -> bool:
    if not (_baseline_scores_valid(baseline_scores) and _score_dicts_valid(score_dicts_by_feature)):
        return False
    return set(baseline_scores.keys()) == set(score_dicts_by_feature[0].keys())

def _multi_metric_bunches_valid(result: object, baseline_scores: Mapping[str, float]) -> bool:
    from sklearn.utils import Bunch, check_random_state
    return bool(
        isinstance(result, dict)
        and set(result.keys()) == set(baseline_scores.keys())
        and all(isinstance(value, Bunch) for value in result.values())
    )

@register_atom(witness_permutation_importance_random_seed)
@icontract.require(lambda random_state=None: _random_state_valid(random_state), "random_state must be None or an integer seed")
@icontract.ensure(lambda result: _int32_seed_valid(result), "derived random seed must be an int32-range integer")
def permutation_importance_random_seed(random_state: int | None = None) -> int:
    from sklearn.utils import Bunch, check_random_state
    """Derive sklearn's per-call permutation random seed from a public random_state."""
    rng = check_random_state(random_state)
    return int(rng.randint(np.iinfo(np.int32).max + 1))

@register_atom(witness_permutation_importance_metric_score_matrix)
@icontract.require(lambda score_dicts_by_feature: _score_dicts_valid(score_dicts_by_feature), "score_dicts_by_feature must be a nonempty tuple of same-key finite score dictionaries")
@icontract.require(lambda score_dicts_by_feature, metric_name: _metric_name_valid(metric_name, score_dicts_by_feature), "metric_name must be present in the score dictionaries")
@icontract.ensure(lambda result, score_dicts_by_feature, metric_name: _metric_score_matrix_valid(result, score_dicts_by_feature, metric_name), "metric score matrix must stack one score vector per feature")
def permutation_importance_metric_score_matrix(
    score_dicts_by_feature: ScoreDictTuple,
    metric_name: str,
) -> NDArray[np.float64]:
    """Unpack sklearn's per-feature multimetric score dicts into one metric matrix."""
    return np.asarray([score_dict[metric_name] for score_dict in score_dicts_by_feature], dtype=np.float64)

@register_atom(witness_permutation_importance_summary_bunch)
@icontract.require(lambda baseline_score: _finite_score(baseline_score), "baseline_score must be finite")
@icontract.require(lambda permuted_scores: _finite_score_matrix(permuted_scores), "permuted_scores must be a finite nonempty 2D matrix")
@icontract.ensure(lambda result, permuted_scores: _summary_bunch_valid(result, permuted_scores), "summary Bunch must contain sklearn's importances arrays")
def permutation_importance_summary_bunch(
    baseline_score: float,
    permuted_scores: NDArray[np.float64],
) -> Bunch:
    from sklearn.utils import Bunch, check_random_state
    """Build sklearn's importances Bunch from one baseline score and one score matrix."""
    scores = np.asarray(permuted_scores, dtype=np.float64)
    importances = permutation_importance_values(float(baseline_score), scores)
    return Bunch(
        importances_mean=permutation_importance_mean(importances),
        importances_std=permutation_importance_std(importances),
        importances=importances,
    )

@register_atom(witness_permutation_importance_multi_metric_bunches)
@icontract.require(lambda baseline_scores, score_dicts_by_feature: _multi_metric_inputs_valid(baseline_scores, score_dicts_by_feature), "baseline_scores and score_dicts_by_feature must share the same nonempty metric keys")
@icontract.ensure(lambda result, baseline_scores: _multi_metric_bunches_valid(result, baseline_scores), "result must be a metric-keyed dict of sklearn importances Bunches")
def permutation_importance_multi_metric_bunches(
    baseline_scores: dict[str, float],
    score_dicts_by_feature: ScoreDictTuple,
) -> dict[str, Bunch]:
    from sklearn.utils import Bunch, check_random_state
    """Build sklearn's multimetric permutation-importance dict from scorer outputs."""
    return {
        metric_name: permutation_importance_summary_bunch(
            float(baseline_scores[metric_name]),
            permutation_importance_metric_score_matrix(score_dicts_by_feature, metric_name),
        )
        for metric_name in baseline_scores
    }
