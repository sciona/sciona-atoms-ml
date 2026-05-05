"""Deterministic sklearn coordinate-descent CV API shell atoms."""

from .atoms import (
    cd_cv_metadata_router_spec,
    cd_cv_multitask_bool,
    cd_cv_sparse_input_tag,
    cd_cv_target_multi_output_tag,
    cd_cv_target_single_output_tag,
)

__all__ = [
    "cd_cv_metadata_router_spec",
    "cd_cv_multitask_bool",
    "cd_cv_sparse_input_tag",
    "cd_cv_target_multi_output_tag",
    "cd_cv_target_single_output_tag",
]
