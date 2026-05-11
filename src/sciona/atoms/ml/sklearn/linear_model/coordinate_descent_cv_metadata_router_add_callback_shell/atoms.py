"""Sklearn LinearModelCV MetadataRouter.add callback atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import witness_cd_cv_metadata_router_add_result


def _fit_split_mapping(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("caller") == "fit"
        and value.get("callee") == "split"
    )


@register_atom(witness_cd_cv_metadata_router_add_result)
@icontract.require(lambda router_after_add: router_after_add is not None, "router result must be present")
@icontract.require(lambda splitter: splitter is not None, "splitter must be present")
@icontract.require(
    lambda method_mapping: _fit_split_mapping(method_mapping),
    "method_mapping must describe fit calling split",
)
@icontract.ensure(
    lambda result, router_after_add: result is router_after_add,
    "MetadataRouter.add result must preserve callback output identity",
)
def cd_cv_metadata_router_add_result(
    router_after_add: object,
    splitter: object,
    method_mapping: Mapping[str, str],
) -> object:
    """Return the MetadataRouter.add(...) callback result before final return."""
    del splitter, method_mapping
    return router_after_add
