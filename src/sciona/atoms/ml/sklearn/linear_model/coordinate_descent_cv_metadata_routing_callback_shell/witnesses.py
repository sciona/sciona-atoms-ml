"""Ghost witnesses for sklearn coordinate-descent CV metadata-routing callback-shell atoms."""

from __future__ import annotations


def witness_cd_cv_splitter_consumes_kwargs(cv: object) -> object:
    """Describe the get_routing_for_object(cv).consumes(...) kwarg shell."""
    return cv


def witness_cd_cv_splitter_supports_sample_weight_result(
    splitter_supports_sample_weight: object,
) -> object:
    """Describe the boolean result from splitter routing consumption lookup."""
    return splitter_supports_sample_weight


def witness_cd_cv_routing_params_with_sample_weight(
    params: object, splitter_supports_sample_weight: object, sample_weight: object
) -> object:
    """Describe params after optional splitter sample_weight forwarding."""
    return params, splitter_supports_sample_weight, sample_weight


def witness_cd_cv_process_routing_args(estimator: object) -> object:
    """Describe positional args for process_routing(self, 'fit', **params)."""
    return estimator


def witness_cd_cv_process_routing_kwargs(params: object) -> object:
    """Describe kwargs expanded into process_routing(self, 'fit', **params)."""
    return params


def witness_cd_cv_routed_params_result(routed_params: object) -> object:
    """Describe routed_params returned by deferred process_routing(...)."""
    return routed_params
