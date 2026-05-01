"""GraphicalLassoCV path postprocessing helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_graphical_lasso_cv_alphas_with_baseline,
    witness_graphical_lasso_cv_best_alpha,
    witness_graphical_lasso_cv_path_alphas,
    witness_graphical_lasso_cv_path_score_matrix,
    witness_graphical_lasso_cv_scores_with_baseline,
    witness_graphical_lasso_cv_sorted_path_records,
)


def _finite_scalar(value: object) -> bool:
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool) and np.isfinite(float(value))


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _descending_positive_unique_vector(values: object) -> bool:
    if not _finite_vector(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(array > 0.0) and np.all(array[:-1] > array[1:]))


def _descending_nonnegative_unique_vector(values: object) -> bool:
    if not _finite_vector(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(array >= 0.0) and np.all(array[:-1] > array[1:]))


def _score_matrix_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _path_records_valid(path_records: object) -> bool:
    if isinstance(path_records, (str, bytes)) or not isinstance(path_records, Sequence):
        return False
    if len(path_records) < 1:
        return False
    fold_count: int | None = None
    for record in path_records:
        if not (isinstance(record, tuple) and len(record) == 3):
            return False
        alpha, scores, _covs = record
        if not _finite_scalar(alpha) or float(alpha) <= 0.0:
            return False
        if not _finite_vector(scores):
            return False
        score_values = np.asarray(scores, dtype=np.float64)
        if fold_count is None:
            fold_count = int(score_values.shape[0])
        elif int(score_values.shape[0]) != fold_count:
            return False
    return True


def _sorted_path_records_valid(result: object) -> bool:
    if not _path_records_valid(result):
        return False
    alphas = np.asarray([float(record[0]) for record in result], dtype=np.float64)
    return bool(np.all(alphas[:-1] >= alphas[1:]))


def _path_alpha_vector_valid(result: object, path_records: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    if not _descending_positive_unique_vector(values):
        return False
    records = tuple(path_records)
    expected = np.asarray([float(record[0]) for record in records], dtype=np.float64)
    return bool(values.shape == expected.shape and np.array_equal(values, expected))


def _path_score_matrix_matches(result: object, path_records: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    if not _score_matrix_valid(values):
        return False
    records = tuple(path_records)
    expected = np.asarray([np.asarray(record[1], dtype=np.float64) for record in records], dtype=np.float64)
    return bool(values.shape == expected.shape and np.allclose(values, expected))


def _baseline_alpha_vector_valid(result: object, alphas: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(alphas, dtype=np.float64)
    return bool(
        _descending_nonnegative_unique_vector(values)
        and values.shape == (source.shape[0] + 1,)
        and np.array_equal(values[:-1], source)
        and float(values[-1]) == 0.0
    )


def _baseline_score_matrix_valid(result: object, grid_scores: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(grid_scores, dtype=np.float64)
    return bool(_score_matrix_valid(values) and values.shape[0] == source.shape[0] + 1 and values.shape[1] == source.shape[1])


def _best_index_valid(best_index: object, alphas: object) -> bool:
    return _integer(best_index) and 0 <= int(best_index) < np.asarray(alphas, dtype=np.float64).shape[0]


@register_atom(witness_graphical_lasso_cv_sorted_path_records)
@icontract.require(lambda path_records: _path_records_valid(path_records), "path_records must be a non-empty sequence of positive-alpha GraphicalLassoCV path tuples with aligned finite score vectors")
@icontract.ensure(lambda result: _sorted_path_records_valid(result), "sorted path records must remain valid and be ordered by descending alpha")
def graphical_lasso_cv_sorted_path_records(
    path_records: object,
) -> tuple[tuple[float, tuple[float, ...], object], ...]:
    """Sort GraphicalLassoCV path records by descending alpha after path extension."""
    records = tuple(path_records)
    return tuple(sorted(records, key=lambda record: float(record[0]), reverse=True))


@register_atom(witness_graphical_lasso_cv_path_alphas)
@icontract.require(lambda path_records: _sorted_path_records_valid(path_records), "path_records must be valid descending-alpha path tuples")
@icontract.ensure(lambda result, path_records: _path_alpha_vector_valid(result, path_records), "path alphas must match the alpha column of path_records")
def graphical_lasso_cv_path_alphas(
    path_records: object,
) -> NDArray[np.float64]:
    """Unpack GraphicalLassoCV's descending alpha vector from sorted path records."""
    records = tuple(path_records)
    return np.asarray([float(record[0]) for record in records], dtype=np.float64)


@register_atom(witness_graphical_lasso_cv_path_score_matrix)
@icontract.require(lambda path_records: _sorted_path_records_valid(path_records), "path_records must be valid descending-alpha path tuples")
@icontract.ensure(lambda result, path_records: _path_score_matrix_matches(result, path_records), "path score matrix must match the score column of path_records")
def graphical_lasso_cv_path_score_matrix(
    path_records: object,
) -> NDArray[np.float64]:
    """Unpack GraphicalLassoCV's alpha-by-fold score matrix from sorted path records."""
    records = tuple(path_records)
    return np.asarray([np.asarray(record[1], dtype=np.float64) for record in records], dtype=np.float64)


@register_atom(witness_graphical_lasso_cv_alphas_with_baseline)
@icontract.require(lambda alphas: _descending_positive_unique_vector(alphas), "alphas must be a strictly decreasing positive vector")
@icontract.ensure(lambda result, alphas: _baseline_alpha_vector_valid(result, alphas), "result must append a terminal zero alpha to the source vector")
def graphical_lasso_cv_alphas_with_baseline(
    alphas: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Append GraphicalLassoCV's empirical covariance baseline alpha of zero."""
    alpha_values = np.asarray(alphas, dtype=np.float64)
    return np.concatenate([alpha_values, np.array([0.0], dtype=np.float64)])


@register_atom(witness_graphical_lasso_cv_scores_with_baseline)
@icontract.require(lambda grid_scores: _score_matrix_valid(grid_scores), "grid_scores must be a finite alpha-by-fold score matrix")
@icontract.require(lambda empirical_scores, grid_scores: _finite_vector(empirical_scores) and np.asarray(empirical_scores, dtype=np.float64).shape[0] == np.asarray(grid_scores, dtype=np.float64).shape[1], "empirical_scores must be a finite per-fold score vector matching the fold dimension of grid_scores")
@icontract.ensure(lambda result, grid_scores: _baseline_score_matrix_valid(result, grid_scores), "result must append one baseline score row to grid_scores")
def graphical_lasso_cv_scores_with_baseline(
    grid_scores: NDArray[np.float64],
    empirical_scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Append GraphicalLassoCV's empirical covariance baseline scores as the final row."""
    score_values = np.asarray(grid_scores, dtype=np.float64)
    empirical_row = np.asarray(empirical_scores, dtype=np.float64).reshape(1, -1)
    return np.vstack([score_values, empirical_row])


@register_atom(witness_graphical_lasso_cv_best_alpha)
@icontract.require(lambda alphas: _descending_nonnegative_unique_vector(alphas), "alphas must be a strictly decreasing nonnegative vector")
@icontract.require(lambda best_index, alphas: _best_index_valid(best_index, alphas), "best_index must index into alphas")
@icontract.ensure(lambda result: _finite_scalar(result), "best alpha must be finite")
def graphical_lasso_cv_best_alpha(
    alphas: NDArray[np.float64],
    best_index: int,
) -> float:
    """Select GraphicalLassoCV's final alpha from the indexed alpha vector."""
    alpha_values = np.asarray(alphas, dtype=np.float64)
    return float(alpha_values[int(best_index)])
