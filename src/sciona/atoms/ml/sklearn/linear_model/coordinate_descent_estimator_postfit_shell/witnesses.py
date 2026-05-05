"""Ghost witnesses for sklearn coordinate-descent estimator post-fit shell atoms."""

from __future__ import annotations


def witness_cd_estimator_single_target_branch(n_targets: object) -> object:
    """Describe the `if n_targets == 1:` shell in ElasticNet.fit."""
    return n_targets


def witness_cd_estimator_single_target_n_iter(n_iter_list: object) -> object:
    """Describe the single-target n_iter collapse in ElasticNet.fit."""
    return n_iter_list


def witness_cd_estimator_single_target_coef(coef_matrix: object) -> object:
    """Describe the single-target coef collapse in ElasticNet.fit."""
    return coef_matrix


def witness_cd_estimator_single_target_dual_gap(dual_gaps: object) -> object:
    """Describe the single-target dual-gap collapse in ElasticNet.fit."""
    return dual_gaps


def witness_cd_estimator_nonfinite_parameter_guard_required(
    coef: object, intercept: object
) -> object:
    """Describe the non-finite parameter guard in ElasticNet.fit."""
    return coef, intercept


def witness_cd_estimator_nonfinite_parameter_message(
    coef: object, intercept: object
) -> object:
    """Describe the non-finite parameter ValueError message in ElasticNet.fit."""
    return coef, intercept


def witness_cd_estimator_sparse_coef(coef: object) -> object:
    """Describe the sparse_coef_ property shell in ElasticNet."""
    return coef
