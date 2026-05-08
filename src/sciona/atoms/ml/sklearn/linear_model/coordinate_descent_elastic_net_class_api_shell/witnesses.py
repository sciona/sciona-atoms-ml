"""Ghost witnesses for sklearn coordinate-descent ElasticNet class-API atoms."""

from __future__ import annotations


def witness_cd_elastic_net_fit_metadata_request(estimator_kind: object) -> object:
    """Describe ElasticNet's class-level fit metadata request."""
    return estimator_kind


def witness_cd_elastic_net_parameter_constraint_names(estimator_kind: object) -> object:
    """Describe ElasticNet parameter-constraint declaration names."""
    return estimator_kind


def witness_cd_elastic_net_parameter_constraint_descriptors(estimator_kind: object) -> object:
    """Describe compact ElasticNet parameter-constraint descriptors."""
    return estimator_kind
