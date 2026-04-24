"""Deterministic RFECV post-fold bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import witness_rfecv_best_feature_count, witness_rfecv_cv_results


def _score_matrix_valid(step_scores: NDArray[np.float64]) -> bool:
    values = np.asarray(step_scores, dtype=np.float64)
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.all(np.isfinite(values))
    )


def _strictly_decreasing(values: NDArray[np.int64]) -> bool:
    return bool(values.shape[0] >= 1 and np.all(values >= 1) and (values.shape[0] == 1 or np.all(np.diff(values) < 0)))


def _integer_path_valid(step_n_features: NDArray[np.int64]) -> bool:
    values = np.asarray(step_n_features)
    return bool(values.ndim in {1, 2} and values.size >= 1 and values.dtype != np.bool_ and np.issubdtype(values.dtype, np.integer))


def _step_feature_path_valid(step_scores: NDArray[np.float64], step_n_features: NDArray[np.int64]) -> bool:
    if not _score_matrix_valid(step_scores) or not _integer_path_valid(step_n_features):
        return False

    scores = np.asarray(step_scores, dtype=np.float64)
    paths = np.asarray(step_n_features)
    n_folds, n_steps = scores.shape

    if paths.ndim == 1:
        return bool(paths.shape == (n_steps,) and _strictly_decreasing(np.asarray(paths, dtype=np.int64)))

    if paths.shape != (n_folds, n_steps):
        return False

    first_row = np.asarray(paths[0], dtype=np.int64)
    if not _strictly_decreasing(first_row):
        return False

    return bool(np.all(paths == first_row))


def _normalized_step_feature_path(step_n_features: NDArray[np.int64]) -> NDArray[np.int64]:
    values = np.asarray(step_n_features, dtype=np.int64)
    path = values if values.ndim == 1 else values[0]
    return np.asarray(path[::-1], dtype=np.int64)


def _cv_results_valid(
    result: dict[str, NDArray[np.float64] | NDArray[np.int64]],
    step_scores: NDArray[np.float64],
) -> bool:
    scores = np.asarray(step_scores, dtype=np.float64)
    n_folds, n_steps = scores.shape
    expected_keys = [f"split{i}_test_score" for i in range(n_folds)]
    expected = {"mean_test_score", "std_test_score", "n_features", *expected_keys}
    if set(result) != expected:
        return False

    mean_scores = np.asarray(result["mean_test_score"], dtype=np.float64)
    std_scores = np.asarray(result["std_test_score"], dtype=np.float64)
    n_features = np.asarray(result["n_features"], dtype=np.int64)
    if mean_scores.shape != (n_steps,) or std_scores.shape != (n_steps,) or n_features.shape != (n_steps,):
        return False
    if not np.all(np.isfinite(mean_scores)) or not np.all(np.isfinite(std_scores)):
        return False

    for index in range(n_folds):
        split_scores = np.asarray(result[f"split{index}_test_score"], dtype=np.float64)
        if split_scores.shape != (n_steps,) or not np.all(np.isfinite(split_scores)):
            return False

    return True


@register_atom(witness_rfecv_best_feature_count)
@icontract.require(lambda step_scores: _score_matrix_valid(step_scores), "step_scores must be a finite fold-by-step score matrix")
@icontract.require(
    lambda step_scores, step_n_features: _step_feature_path_valid(step_scores, step_n_features),
    "step_n_features must be a shared strictly decreasing elimination path matching the score matrix",
)
@icontract.ensure(
    lambda result, step_n_features: isinstance(result, int) and result in set(np.asarray(step_n_features, dtype=np.int64).ravel()),
    "best feature count must come from the supplied elimination path",
)
def rfecv_best_feature_count(
    step_scores: NDArray[np.float64],
    step_n_features: NDArray[np.int64],
) -> int:
    """Choose the RFECV feature count using sklearn's reversed tie-break path."""
    scores = np.asarray(step_scores, dtype=np.float64)
    step_n_features_rev = _normalized_step_feature_path(step_n_features)
    scores_sum_rev = np.sum(scores, axis=0)[::-1]
    return int(step_n_features_rev[np.argmax(scores_sum_rev)])


@register_atom(witness_rfecv_cv_results)
@icontract.require(lambda step_scores: _score_matrix_valid(step_scores), "step_scores must be a finite fold-by-step score matrix")
@icontract.require(
    lambda step_scores, step_n_features: _step_feature_path_valid(step_scores, step_n_features),
    "step_n_features must be a shared strictly decreasing elimination path matching the score matrix",
)
@icontract.ensure(
    lambda result, step_scores: _cv_results_valid(result, step_scores),
    "cv results must contain mean, std, per-split scores, and reversed feature counts",
)
def rfecv_cv_results(
    step_scores: NDArray[np.float64],
    step_n_features: NDArray[np.int64],
) -> dict[str, NDArray[np.float64] | NDArray[np.int64]]:
    """Materialize RFECV cv_results_ from per-fold step scores and feature counts."""
    scores = np.asarray(step_scores, dtype=np.float64)
    step_n_features_rev = _normalized_step_feature_path(step_n_features)
    scores_rev = scores[:, ::-1]
    return {
        "mean_test_score": np.asarray(np.mean(scores_rev, axis=0), dtype=np.float64),
        "std_test_score": np.asarray(np.std(scores_rev, axis=0), dtype=np.float64),
        **{f"split{i}_test_score": np.asarray(scores_rev[i], dtype=np.float64) for i in range(scores.shape[0])},
        "n_features": step_n_features_rev,
    }
