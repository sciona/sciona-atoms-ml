"""Ghost witnesses for sklearn coordinate-descent CV non-routing fallback-shell atoms."""

from __future__ import annotations


def witness_cd_cv_nonrouting_empty_split_params(default_routed_params_required: object) -> object:
    """Describe the empty Bunch used as routed_params.splitter.split."""
    return default_routed_params_required


def witness_cd_cv_nonrouting_splitter_payload(split_params: object) -> object:
    """Describe the Bunch(split=...) fallback payload."""
    return split_params


def witness_cd_cv_nonrouting_routed_params(
    default_routed_params_required: object, splitter_payload: object
) -> object:
    """Describe the routed_params Bunch with a splitter attribute."""
    return default_routed_params_required, splitter_payload


def witness_cd_cv_nonrouting_split_kwargs(routed_params: object) -> object:
    """Describe extraction of routed_params.splitter.split for cv.split kwargs."""
    return routed_params
