"""Ghost witnesses for sklearn RANSAC metadata-routing atoms."""

from __future__ import annotations


def witness_ransac_metadata_router_owner(class_name: str) -> str:
    """Describe MetadataRouter owner name resolution."""
    return class_name


def witness_ransac_metadata_method_mapping_add_kwargs(caller: str, callee: str) -> object:
    """Describe MethodMapping.add kwargs for one RANSAC route."""
    return (caller, callee)


def witness_ransac_metadata_estimator_payload(estimator: object, method_mapping: object) -> object:
    """Describe MetadataRouter.add payload for the wrapped estimator."""
    return (estimator, method_mapping)


def witness_ransac_metadata_router_result(router: object) -> object:
    """Describe final MetadataRouter return identity."""
    return router
