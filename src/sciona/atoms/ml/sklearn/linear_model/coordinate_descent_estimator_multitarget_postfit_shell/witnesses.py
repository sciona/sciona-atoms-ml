"""Ghost witnesses for sklearn ElasticNet multi-target post-fit atoms."""

from __future__ import annotations


def witness_cd_estimator_multitarget_branch(n_targets: object) -> object:
    """Describe whether ElasticNet.fit keeps multi-target outputs as arrays."""
    return n_targets


def witness_cd_estimator_multitarget_coef(coef_matrix: object, n_targets: object) -> object:
    """Describe multi-target coefficient passthrough."""
    return coef_matrix, n_targets


def witness_cd_estimator_multitarget_dual_gap(dual_gaps: object, n_targets: object) -> object:
    """Describe multi-target dual-gap passthrough."""
    return dual_gaps, n_targets
