"""Ghost witnesses for sklearn LinearModelCV metadata-router MethodMapping callback atoms."""

from __future__ import annotations


def witness_cd_cv_metadata_router_method_mapping_add_kwargs(
    caller: object,
    callee: object,
) -> object:
    """Describe MethodMapping().add(caller='fit', callee='split') kwargs."""
    return caller, callee


def witness_cd_cv_metadata_router_method_mapping_result(method_mapping: object) -> object:
    """Describe the MethodMapping.add(...) result passed into MetadataRouter.add."""
    return method_mapping
