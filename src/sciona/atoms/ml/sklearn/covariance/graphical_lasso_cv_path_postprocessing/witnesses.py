"""Ghost witnesses for GraphicalLassoCV path postprocessing helper atoms."""

from __future__ import annotations

from numpy.typing import NDArray


def witness_graphical_lasso_cv_sorted_path_records(
    path_records: object,
) -> tuple[tuple[float, tuple[float, ...], object], ...]:
    """Describe GraphicalLassoCV path records sorted by descending alpha."""
    del path_records
    raise NotImplementedError


def witness_graphical_lasso_cv_path_alphas(
    path_records: object,
) -> NDArray[float]:
    """Describe the alpha vector unpacked from GraphicalLassoCV path records."""
    del path_records
    raise NotImplementedError


def witness_graphical_lasso_cv_path_score_matrix(
    path_records: object,
) -> NDArray[float]:
    """Describe the alpha-by-fold score matrix unpacked from GraphicalLassoCV path records."""
    del path_records
    raise NotImplementedError


def witness_graphical_lasso_cv_alphas_with_baseline(
    alphas: NDArray[float],
) -> NDArray[float]:
    """Describe GraphicalLassoCV's alpha vector after appending the empirical baseline alpha of zero."""
    del alphas
    raise NotImplementedError


def witness_graphical_lasso_cv_scores_with_baseline(
    grid_scores: NDArray[float],
    empirical_scores: NDArray[float],
) -> NDArray[float]:
    """Describe GraphicalLassoCV's score matrix after appending empirical covariance baseline scores."""
    del grid_scores
    del empirical_scores
    raise NotImplementedError


def witness_graphical_lasso_cv_best_alpha(
    alphas: NDArray[float],
    best_index: int,
) -> float:
    """Describe the selected GraphicalLassoCV alpha from an index into the final alpha vector."""
    del alphas
    del best_index
    return 0.0
