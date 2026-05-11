"""Ghost witnesses for sklearn LinearModelCV MetadataRouter.add callback atoms."""

from __future__ import annotations


def witness_cd_cv_metadata_router_add_result(
    router_after_add: object,
    splitter: object,
    method_mapping: object,
) -> object:
    """Describe the MetadataRouter.add(...) callback output."""
    return router_after_add, splitter, method_mapping
