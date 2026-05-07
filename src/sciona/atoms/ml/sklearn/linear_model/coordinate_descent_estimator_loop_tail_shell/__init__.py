"""Deterministic sklearn coordinate-descent estimator loop-tail atoms."""

from .atoms import (
    cd_estimator_coef_matrix_with_target,
    cd_estimator_dual_gaps_with_target,
    cd_estimator_n_iter_with_target,
    cd_estimator_target_coef_column,
    cd_estimator_target_dual_gap_scalar,
    cd_estimator_target_iteration_count,
)

__all__ = [
    "cd_estimator_target_coef_column",
    "cd_estimator_coef_matrix_with_target",
    "cd_estimator_target_dual_gap_scalar",
    "cd_estimator_dual_gaps_with_target",
    "cd_estimator_target_iteration_count",
    "cd_estimator_n_iter_with_target",
]
