"""Ghost witnesses for sklearn LinearModelCV metadata-router payload atoms."""

from __future__ import annotations


def witness_cd_cv_metadata_router_self_request(estimator: object) -> object:
    """Describe the estimator passed to add_self_request(...)."""
    return estimator


def witness_cd_cv_metadata_router_splitter_payload(
    splitter: object,
    method_mapping: object,
) -> object:
    """Describe the splitter payload passed to MetadataRouter.add(...)."""
    return splitter, method_mapping


def witness_cd_cv_metadata_router_result(router: object) -> object:
    """Describe the final router object returned by get_metadata_routing."""
    return router
