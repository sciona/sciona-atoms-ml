"""Permutation output-branching shell atoms adapted from scikit-learn."""

from .atoms import (
    permutation_importance_metric_names,
    permutation_importance_single_metric_score_matrix,
    permutation_importance_use_multimetric_results,
)

__all__ = [
    "permutation_importance_use_multimetric_results",
    "permutation_importance_metric_names",
    "permutation_importance_single_metric_score_matrix",
]
