"""Ghost witnesses for sklearn LinearModelCV parameter-constraint atoms."""

from __future__ import annotations


def witness_cd_cv_base_parameter_constraint_names(estimator_kind: object) -> object:
    """Describe LinearModelCV._parameter_constraints declaration order."""
    return estimator_kind


def witness_cd_cv_base_parameter_constraint_descriptors(estimator_kind: object) -> object:
    """Describe compact descriptors for LinearModelCV._parameter_constraints."""
    return estimator_kind
