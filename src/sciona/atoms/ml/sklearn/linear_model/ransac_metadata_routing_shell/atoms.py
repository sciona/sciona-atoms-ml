"""Sklearn RANSAC metadata-routing atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_ransac_metadata_estimator_payload,
    witness_ransac_metadata_method_mapping_add_kwargs,
    witness_ransac_metadata_router_owner,
    witness_ransac_metadata_router_result,
)

_METHOD_MAPPING_PAIRS = (
    ("fit", "fit"),
    ("fit", "score"),
    ("score", "score"),
    ("predict", "predict"),
)
_ESTIMATOR_PAYLOAD_KEYS = {"estimator", "method_mapping"}


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and len(value) >= 1)


def _method_pairs(value: object) -> bool:
    try:
        pairs = tuple(tuple(pair) for pair in value)  # type: ignore[arg-type]
    except TypeError:
        return False
    return pairs == _METHOD_MAPPING_PAIRS


def _mapping_pair(value: object) -> bool:
    return bool(isinstance(value, Mapping) and (value.get("caller"), value.get("callee")) in _METHOD_MAPPING_PAIRS)


def _owner_valid(result: str, class_name: str) -> bool:
    return bool(result == class_name)


def _add_kwargs_valid(result: dict[str, str], caller: str, callee: str) -> bool:
    return bool(isinstance(result, dict) and result == {"caller": caller, "callee": callee})


def _estimator_payload_valid(result: dict[str, object], estimator: object, method_mapping: object) -> bool:
    return bool(
        set(result) == _ESTIMATOR_PAYLOAD_KEYS
        and result["estimator"] is estimator
        and result["method_mapping"] is method_mapping
    )


@register_atom(witness_ransac_metadata_router_owner)
@icontract.require(lambda class_name: _nonempty_string(class_name), "class_name must be nonempty")
@icontract.ensure(lambda result, class_name: _owner_valid(result, class_name), "owner must match self.__class__.__name__")
def ransac_metadata_router_owner(class_name: str) -> str:
    """Return the MetadataRouter owner name for RANSACRegressor."""
    return class_name


@register_atom(witness_ransac_metadata_method_mapping_add_kwargs)
@icontract.require(lambda caller, callee: (caller, callee) in _METHOD_MAPPING_PAIRS, "caller/callee pair must be one of the RANSAC routes")
@icontract.ensure(lambda result, caller, callee: _add_kwargs_valid(result, caller, callee), "MethodMapping.add kwargs must preserve caller and callee")
def ransac_metadata_method_mapping_add_kwargs(caller: str, callee: str) -> dict[str, str]:
    """Return kwargs for one MethodMapping.add call in RANSACRegressor."""
    return {"caller": caller, "callee": callee}


@register_atom(witness_ransac_metadata_estimator_payload)
@icontract.require(lambda method_mapping: _method_pairs(method_mapping) or _mapping_pair(method_mapping), "method_mapping must describe RANSAC estimator routes")
@icontract.ensure(
    lambda result, estimator, method_mapping: _estimator_payload_valid(result, estimator, method_mapping),
    "MetadataRouter.add payload must preserve estimator and method mapping identities",
)
def ransac_metadata_estimator_payload(estimator: object, method_mapping: object) -> dict[str, object]:
    """Return the named payload passed to MetadataRouter.add(...)."""
    return {"estimator": estimator, "method_mapping": method_mapping}


@register_atom(witness_ransac_metadata_router_result)
@icontract.require(lambda router: router is not None, "router must be present")
@icontract.ensure(lambda result, router: result is router, "get_metadata_routing must return the constructed router")
def ransac_metadata_router_result(router: object) -> object:
    """Return the constructed MetadataRouter from get_metadata_routing."""
    return router
