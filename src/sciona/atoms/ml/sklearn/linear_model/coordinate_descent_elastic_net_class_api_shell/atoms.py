"""Sklearn coordinate-descent ElasticNet class-API atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_elastic_net_fit_metadata_request,
    witness_cd_elastic_net_parameter_constraint_descriptors,
    witness_cd_elastic_net_parameter_constraint_names,
)


_CONSTRAINT_NAMES = (
    "alpha",
    "l1_ratio",
    "fit_intercept",
    "precompute",
    "max_iter",
    "copy_X",
    "tol",
    "warm_start",
    "positive",
    "random_state",
    "selection",
)

_CONSTRAINT_DESCRIPTORS = {
    "alpha": (("interval", "Real", 0, None, "left"),),
    "l1_ratio": (("interval", "Real", 0, 1, "both"),),
    "fit_intercept": ("boolean",),
    "precompute": ("boolean", "array-like"),
    "max_iter": (("interval", "Integral", 1, None, "left"), None),
    "copy_X": ("boolean",),
    "tol": (("interval", "Real", 0, None, "left"),),
    "warm_start": ("boolean",),
    "positive": ("boolean",),
    "random_state": ("random_state",),
    "selection": (("str_options", ("cyclic", "random")),),
}


@register_atom(witness_cd_elastic_net_fit_metadata_request)
@icontract.require(lambda estimator_kind: estimator_kind == "elastic_net", "estimator_kind must be elastic_net")
@icontract.ensure(
    lambda result: result == {"check_input": "UNUSED"},
    "ElasticNet fit metadata request must mark check_input as UNUSED",
)
def cd_elastic_net_fit_metadata_request(estimator_kind: str) -> dict[str, str]:
    """Return the class-level fit metadata request for ElasticNet."""
    del estimator_kind
    return {"check_input": "UNUSED"}


@register_atom(witness_cd_elastic_net_parameter_constraint_names)
@icontract.require(lambda estimator_kind: estimator_kind == "elastic_net", "estimator_kind must be elastic_net")
@icontract.ensure(
    lambda result: isinstance(result, tuple) and result == _CONSTRAINT_NAMES,
    "ElasticNet parameter constraint names must preserve sklearn declaration order",
)
def cd_elastic_net_parameter_constraint_names(estimator_kind: str) -> tuple[str, ...]:
    """Return ElasticNet._parameter_constraints names in declaration order."""
    del estimator_kind
    return _CONSTRAINT_NAMES


@register_atom(witness_cd_elastic_net_parameter_constraint_descriptors)
@icontract.require(lambda estimator_kind: estimator_kind == "elastic_net", "estimator_kind must be elastic_net")
@icontract.ensure(
    lambda result: isinstance(result, dict)
    and tuple(result) == _CONSTRAINT_NAMES
    and result == _CONSTRAINT_DESCRIPTORS,
    "ElasticNet parameter constraint descriptors must match the class declaration",
)
def cd_elastic_net_parameter_constraint_descriptors(estimator_kind: str) -> dict[str, tuple[object, ...]]:
    """Return compact descriptors for ElasticNet._parameter_constraints."""
    del estimator_kind
    return dict(_CONSTRAINT_DESCRIPTORS)
