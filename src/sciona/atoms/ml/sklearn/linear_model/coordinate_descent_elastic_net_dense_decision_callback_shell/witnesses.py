"""Ghost witnesses for sklearn ElasticNet dense decision callback atoms."""

from __future__ import annotations


def witness_cd_elastic_net_check_is_fitted_args(estimator: object) -> object:
    """Describe the check_is_fitted(self) callback payload."""
    return estimator


def witness_cd_elastic_net_dense_decision_required(is_sparse: object) -> object:
    """Describe the dense decision branch predicate."""
    return is_sparse


def witness_cd_elastic_net_dense_super_decision_args(estimator: object, X: object) -> object:
    """Describe the dense superclass decision callback payload."""
    return estimator, X


def witness_cd_elastic_net_dense_super_decision_result(decision_result: object) -> object:
    """Describe the dense superclass decision callback result."""
    return decision_result
