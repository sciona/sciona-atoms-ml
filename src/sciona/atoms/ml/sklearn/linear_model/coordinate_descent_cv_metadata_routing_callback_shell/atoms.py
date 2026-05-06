"""Sklearn coordinate-descent CV metadata-routing callback-shell atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_process_routing_args,
    witness_cd_cv_process_routing_kwargs,
    witness_cd_cv_routed_params_result,
    witness_cd_cv_routing_params_with_sample_weight,
    witness_cd_cv_splitter_consumes_kwargs,
    witness_cd_cv_splitter_supports_sample_weight_result,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_cd_cv_splitter_consumes_kwargs)
@icontract.ensure(
    lambda result: isinstance(result, dict)
    and result == {"method": "split", "params": ["sample_weight"]},
    "splitter consumes kwargs must match sklearn's metadata-routing query",
)
def cd_cv_splitter_consumes_kwargs(cv: object) -> dict[str, object]:
    """Return kwargs for get_routing_for_object(cv).consumes(...)."""
    del cv
    return {"method": "split", "params": ["sample_weight"]}


@register_atom(witness_cd_cv_splitter_supports_sample_weight_result)
@icontract.require(
    lambda splitter_supports_sample_weight: _bool(splitter_supports_sample_weight),
    "splitter_supports_sample_weight must be boolean",
)
@icontract.ensure(
    lambda result, splitter_supports_sample_weight: _bool(result)
    and result == splitter_supports_sample_weight,
    "splitter sample-weight support result must pass through unchanged",
)
def cd_cv_splitter_supports_sample_weight_result(
    splitter_supports_sample_weight: bool,
) -> bool:
    """Return the boolean result from the deferred consumes(...) callback."""
    return splitter_supports_sample_weight


@register_atom(witness_cd_cv_routing_params_with_sample_weight)
@icontract.require(lambda params: isinstance(params, dict), "params must be a dict")
@icontract.require(
    lambda splitter_supports_sample_weight: _bool(splitter_supports_sample_weight),
    "splitter_supports_sample_weight must be boolean",
)
@icontract.ensure(
    lambda result, params, splitter_supports_sample_weight, sample_weight: isinstance(result, dict)
    and all(result[key] == value for key, value in params.items())
    and (
        result.get("sample_weight") is sample_weight
        if splitter_supports_sample_weight
        else result == params
    ),
    "routing params must add sample_weight only when the splitter consumes it",
)
def cd_cv_routing_params_with_sample_weight(
    params: dict[object, object],
    splitter_supports_sample_weight: bool,
    sample_weight: object,
) -> dict[object, object]:
    """Return params after sklearn's splitter sample_weight forwarding branch."""
    result = dict(params)
    if splitter_supports_sample_weight:
        result["sample_weight"] = sample_weight
    return result


@register_atom(witness_cd_cv_process_routing_args)
@icontract.ensure(
    lambda result, estimator: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is estimator
    and result[1] == "fit",
    "process_routing positional args must preserve estimator and fixed method",
)
def cd_cv_process_routing_args(estimator: object) -> tuple[object, str]:
    """Return positional args for process_routing(self, 'fit', **params)."""
    return (estimator, "fit")


@register_atom(witness_cd_cv_process_routing_kwargs)
@icontract.require(lambda params: isinstance(params, dict), "params must be a dict")
@icontract.ensure(
    lambda result, params: isinstance(result, dict) and result == params,
    "process_routing kwargs must preserve params",
)
def cd_cv_process_routing_kwargs(params: dict[object, object]) -> dict[object, object]:
    """Return kwargs expanded into process_routing(self, 'fit', **params)."""
    return dict(params)


@register_atom(witness_cd_cv_routed_params_result)
@icontract.ensure(
    lambda result, routed_params: result is routed_params,
    "process_routing callback shell must preserve routed_params identity",
)
def cd_cv_routed_params_result(routed_params: object) -> object:
    """Return the routed_params object from the deferred process_routing callback."""
    return routed_params
