"""Sklearn LinearModelCV metadata-router payload atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_metadata_router_result,
    witness_cd_cv_metadata_router_self_request,
    witness_cd_cv_metadata_router_splitter_payload,
)


def _fit_split_mapping(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("caller") == "fit"
        and value.get("callee") == "split"
    )


@register_atom(witness_cd_cv_metadata_router_self_request)
@icontract.ensure(
    lambda result, estimator: result is estimator,
    "add_self_request payload must preserve estimator identity",
)
def cd_cv_metadata_router_self_request(estimator: object) -> object:
    """Return the estimator passed to MetadataRouter.add_self_request(...)."""
    return estimator


@register_atom(witness_cd_cv_metadata_router_splitter_payload)
@icontract.require(
    lambda method_mapping: _fit_split_mapping(method_mapping),
    "method_mapping must describe fit calling split",
)
@icontract.ensure(
    lambda result, splitter, method_mapping: isinstance(result, dict)
    and set(result) == {"splitter", "method_mapping"}
    and result["splitter"] is splitter
    and result["method_mapping"] is method_mapping,
    "MetadataRouter.add payload must preserve splitter and method mapping identity",
)
def cd_cv_metadata_router_splitter_payload(
    splitter: object,
    method_mapping: Mapping[str, str],
) -> dict[str, object]:
    """Return the named payload passed to MetadataRouter.add(...)."""
    return {"splitter": splitter, "method_mapping": method_mapping}


@register_atom(witness_cd_cv_metadata_router_result)
@icontract.ensure(
    lambda result, router: result is router,
    "get_metadata_routing must return the constructed router unchanged",
)
def cd_cv_metadata_router_result(router: object) -> object:
    """Return the constructed MetadataRouter from get_metadata_routing."""
    return router
