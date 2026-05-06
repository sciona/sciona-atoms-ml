"""Deterministic sklearn coordinate-descent CV non-routing fallback-shell atoms."""

from .atoms import (
    cd_cv_nonrouting_empty_split_params,
    cd_cv_nonrouting_routed_params,
    cd_cv_nonrouting_split_kwargs,
    cd_cv_nonrouting_splitter_payload,
)

__all__ = [
    "cd_cv_nonrouting_empty_split_params",
    "cd_cv_nonrouting_splitter_payload",
    "cd_cv_nonrouting_routed_params",
    "cd_cv_nonrouting_split_kwargs",
]
