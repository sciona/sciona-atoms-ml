"""Ghost witnesses for GraphicalLassoCV bookkeeping helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_square(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows = int(values.shape[0])
    cols = int(values.shape[1])
    if rows < 2 or cols < 2 or rows != cols:
        raise ValueError(f"{name} must be square with at least two features")
    return rows


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    length = int(values.shape[0])
    if length < 1:
        raise ValueError(f"{name} must be nonempty")
    return length


def _check_score_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows = int(values.shape[0])
    cols = int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_graphical_lasso_cv_alpha_grid(
    emp_cov: AbstractArray,
    *,
    n_alphas: int,
    alpha_min_ratio: float = 1e-2,
) -> AbstractArray:
    """Describe GraphicalLassoCV's initial alpha grid."""
    _check_square(emp_cov, "emp_cov")
    if n_alphas < 2:
        raise ValueError("n_alphas must be at least 2")
    if not 0.0 < alpha_min_ratio < 1.0:
        raise ValueError("alpha_min_ratio must lie between zero and one")
    return AbstractArray(shape=(n_alphas,), dtype="float64")


def witness_graphical_lasso_cv_mean_test_scores(
    grid_scores: AbstractArray,
    *,
    score_overflow_threshold: float = 0.0,
) -> AbstractArray:
    """Describe GraphicalLassoCV's per-alpha mean score vector."""
    rows, _ = _check_score_matrix(grid_scores, "grid_scores")
    del score_overflow_threshold
    return AbstractArray(shape=(rows,), dtype="float64")


def witness_graphical_lasso_cv_best_index(
    mean_test_scores: AbstractArray,
) -> int:
    """Describe GraphicalLassoCV's selected best-alpha index."""
    _check_vector(mean_test_scores, "mean_test_scores")
    return 0


def witness_graphical_lasso_cv_refinement_bounds(
    alphas: AbstractArray,
    mean_test_scores: AbstractArray,
) -> tuple[float, float]:
    """Describe GraphicalLassoCV's next alpha refinement interval."""
    if _check_vector(alphas, "alphas") != _check_vector(mean_test_scores, "mean_test_scores"):
        raise ValueError("alphas and mean_test_scores must have matching lengths")
    return 1.0, 0.5


def witness_graphical_lasso_cv_refined_alpha_grid(
    alpha_upper: float,
    alpha_lower: float,
    *,
    n_alphas: int,
) -> AbstractArray:
    """Describe GraphicalLassoCV's refined interior alpha grid."""
    if n_alphas < 2:
        raise ValueError("n_alphas must be at least 2")
    if not alpha_upper > alpha_lower > 0.0:
        raise ValueError("alpha bounds must be positive and descending")
    return AbstractArray(shape=(n_alphas,), dtype="float64")


def witness_graphical_lasso_cv_results(
    alphas: AbstractArray,
    grid_scores: AbstractArray,
) -> AbstractArray:
    """Describe GraphicalLassoCV's cv_results_ dictionary materialization."""
    rows = _check_vector(alphas, "alphas")
    score_rows, _ = _check_score_matrix(grid_scores, "grid_scores")
    if rows != score_rows:
        raise ValueError("alphas and grid_scores must have matching alpha rows")
    return AbstractArray(shape=(), dtype="object")
