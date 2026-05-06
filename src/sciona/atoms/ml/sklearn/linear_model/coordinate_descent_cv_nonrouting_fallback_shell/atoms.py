"""Sklearn coordinate-descent CV non-routing fallback-shell atoms."""

from __future__ import annotations

import icontract
from sklearn.utils import Bunch

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_nonrouting_empty_split_params,
    witness_cd_cv_nonrouting_routed_params,
    witness_cd_cv_nonrouting_split_kwargs,
    witness_cd_cv_nonrouting_splitter_payload,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _empty_bunch(value: object) -> bool:
    return isinstance(value, Bunch) and dict(value) == {}


def _has_splitter_split(value: object) -> bool:
    try:
        splitter = getattr(value, "splitter")
    except AttributeError:
        return False
    return hasattr(splitter, "split")


@register_atom(witness_cd_cv_nonrouting_empty_split_params)
@icontract.require(
    lambda default_routed_params_required: _bool(default_routed_params_required)
    and default_routed_params_required,
    "default routed-params fallback must be required",
)
@icontract.ensure(
    lambda result: _empty_bunch(result),
    "non-routing split params must be an empty Bunch",
)
def cd_cv_nonrouting_empty_split_params(default_routed_params_required: bool) -> Bunch:
    """Return sklearn's empty split-params Bunch for the non-routing branch."""
    del default_routed_params_required
    return Bunch()


@register_atom(witness_cd_cv_nonrouting_splitter_payload)
@icontract.require(lambda split_params: isinstance(split_params, Bunch), "split_params must be a Bunch")
@icontract.ensure(
    lambda result, split_params: isinstance(result, Bunch)
    and result.split is split_params,
    "splitter fallback must expose the split Bunch by identity",
)
def cd_cv_nonrouting_splitter_payload(split_params: Bunch) -> Bunch:
    """Return sklearn's Bunch(split=Bunch()) splitter fallback payload."""
    return Bunch(split=split_params)


@register_atom(witness_cd_cv_nonrouting_routed_params)
@icontract.require(
    lambda default_routed_params_required: _bool(default_routed_params_required)
    and default_routed_params_required,
    "default routed-params fallback must be required",
)
@icontract.require(
    lambda splitter_payload: isinstance(splitter_payload, Bunch),
    "splitter_payload must be a Bunch",
)
@icontract.ensure(
    lambda result, splitter_payload: isinstance(result, Bunch)
    and result.splitter is splitter_payload,
    "routed params fallback must expose the splitter payload by identity",
)
def cd_cv_nonrouting_routed_params(
    default_routed_params_required: bool, splitter_payload: Bunch
) -> Bunch:
    """Return sklearn's non-routing routed_params Bunch payload."""
    del default_routed_params_required
    routed_params = Bunch()
    routed_params.splitter = splitter_payload
    return routed_params


@register_atom(witness_cd_cv_nonrouting_split_kwargs)
@icontract.require(
    lambda routed_params: _has_splitter_split(routed_params),
    "routed_params must expose splitter.split",
)
@icontract.ensure(
    lambda result, routed_params: result is routed_params.splitter.split,
    "split kwargs extraction must preserve routed_params.splitter.split identity",
)
def cd_cv_nonrouting_split_kwargs(routed_params: object) -> object:
    """Return the split kwargs payload consumed by cv.split in the fallback branch."""
    return routed_params.splitter.split
