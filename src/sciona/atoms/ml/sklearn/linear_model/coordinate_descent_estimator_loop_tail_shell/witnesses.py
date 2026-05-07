"""Ghost witnesses for sklearn coordinate-descent estimator loop-tail atoms."""

from __future__ import annotations


def witness_cd_estimator_target_coef_column(this_coef: object) -> object:
    """Describe extracting the target coefficient column from path output."""
    return this_coef


def witness_cd_estimator_coef_matrix_with_target(
    coef_matrix: object, target_index: object, this_coef: object
) -> object:
    """Describe writing one target coefficient row into the coefficient matrix."""
    return coef_matrix, target_index, this_coef


def witness_cd_estimator_target_dual_gap_scalar(this_dual_gap: object) -> object:
    """Describe extracting the scalar dual gap from path output."""
    return this_dual_gap


def witness_cd_estimator_dual_gaps_with_target(
    dual_gaps: object, target_index: object, this_dual_gap: object
) -> object:
    """Describe writing one target dual gap into the dual-gap vector."""
    return dual_gaps, target_index, this_dual_gap


def witness_cd_estimator_target_iteration_count(this_iter: object) -> object:
    """Describe extracting the scalar iteration count from path output."""
    return this_iter


def witness_cd_estimator_n_iter_with_target(
    n_iter_list: object, this_iter: object
) -> object:
    """Describe appending one target iteration count."""
    return n_iter_list, this_iter
