"""Deterministic sklearn coordinate-descent CV metadata-routing callback-shell atoms."""

from .atoms import (
    cd_cv_process_routing_args,
    cd_cv_process_routing_kwargs,
    cd_cv_routed_params_result,
    cd_cv_routing_params_with_sample_weight,
    cd_cv_splitter_consumes_kwargs,
    cd_cv_splitter_supports_sample_weight_result,
)

__all__ = [
    "cd_cv_splitter_consumes_kwargs",
    "cd_cv_splitter_supports_sample_weight_result",
    "cd_cv_routing_params_with_sample_weight",
    "cd_cv_process_routing_args",
    "cd_cv_process_routing_kwargs",
    "cd_cv_routed_params_result",
]
