"""Sklearn LinearModelCV metadata-router MethodMapping callback atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_metadata_router_method_mapping_add_kwargs,
    witness_cd_cv_metadata_router_method_mapping_result,
)


def _fit_split_mapping(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("caller") == "fit"
        and value.get("callee") == "split"
    )


@register_atom(witness_cd_cv_metadata_router_method_mapping_add_kwargs)
@icontract.require(lambda caller: caller == "fit", "caller must be fit")
@icontract.require(lambda callee: callee == "split", "callee must be split")
@icontract.ensure(
    lambda result, caller, callee: isinstance(result, dict)
    and result == {"caller": caller, "callee": callee},
    "MethodMapping.add kwargs must describe fit calling split",
)
def cd_cv_metadata_router_method_mapping_add_kwargs(
    caller: str,
    callee: str,
) -> dict[str, str]:
    """Return kwargs for MethodMapping().add(caller='fit', callee='split')."""
    return {"caller": caller, "callee": callee}


@register_atom(witness_cd_cv_metadata_router_method_mapping_result)
@icontract.require(
    lambda method_mapping: _fit_split_mapping(method_mapping),
    "method_mapping must describe fit calling split",
)
@icontract.ensure(
    lambda result, method_mapping: result is method_mapping,
    "MethodMapping.add result must preserve callback output identity",
)
def cd_cv_metadata_router_method_mapping_result(
    method_mapping: Mapping[str, str],
) -> Mapping[str, str]:
    """Return the MethodMapping.add(...) result used by MetadataRouter.add."""
    return method_mapping
