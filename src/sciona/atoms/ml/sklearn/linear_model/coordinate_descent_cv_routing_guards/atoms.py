"""Sklearn coordinate-descent CV routing-guard atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_default_routed_params_required,
    witness_cd_cv_drop_estimator_sample_weight,
    witness_cd_cv_forward_splitter_sample_weight,
    witness_cd_cv_routing_enabled_branch,
    witness_cd_cv_sample_weight_support_guard_required,
    witness_cd_cv_sample_weight_support_message,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_cd_cv_routing_enabled_branch)
@icontract.require(lambda routing_enabled: _bool(routing_enabled), "routing_enabled must be boolean")
@icontract.ensure(
    lambda result, routing_enabled: _bool(result) and result == routing_enabled,
    "routing branch must match _routing_enabled()",
)
def cd_cv_routing_enabled_branch(routing_enabled: bool) -> bool:
    """Return whether LinearModelCV.fit uses metadata routing."""
    return routing_enabled


@register_atom(witness_cd_cv_sample_weight_support_guard_required)
@icontract.require(
    lambda sample_weight_is_not_none: _bool(sample_weight_is_not_none),
    "sample_weight_is_not_none must be boolean",
)
@icontract.require(
    lambda splitter_supports_sample_weight: _bool(splitter_supports_sample_weight),
    "splitter_supports_sample_weight must be boolean",
)
@icontract.require(
    lambda estimator_supports_sample_weight: _bool(estimator_supports_sample_weight),
    "estimator_supports_sample_weight must be boolean",
)
@icontract.ensure(
    lambda result, sample_weight_is_not_none, splitter_supports_sample_weight, estimator_supports_sample_weight: _bool(result)
    and result
    == (
        sample_weight_is_not_none
        and (not splitter_supports_sample_weight)
        and (not estimator_supports_sample_weight)
    ),
    "sample-weight support guard must match sklearn routing logic",
)
def cd_cv_sample_weight_support_guard_required(
    sample_weight_is_not_none: bool,
    splitter_supports_sample_weight: bool,
    estimator_supports_sample_weight: bool,
) -> bool:
    """Return whether LinearModelCV.fit should reject sample weights under routing."""
    return (
        sample_weight_is_not_none
        and (not splitter_supports_sample_weight)
        and (not estimator_supports_sample_weight)
    )


@register_atom(witness_cd_cv_sample_weight_support_message)
@icontract.require(lambda guard_required: _bool(guard_required), "guard_required must be boolean")
@icontract.ensure(
    lambda result: isinstance(result, str)
    and result == "The CV splitter and underlying estimator do not support sample weights.",
    "sample-weight support message must match sklearn formatting",
)
def cd_cv_sample_weight_support_message(guard_required: bool) -> str:
    """Return the unsupported sample-weight ValueError text used by LinearModelCV.fit."""
    del guard_required
    return "The CV splitter and underlying estimator do not support sample weights."


@register_atom(witness_cd_cv_forward_splitter_sample_weight)
@icontract.require(
    lambda splitter_supports_sample_weight: _bool(splitter_supports_sample_weight),
    "splitter_supports_sample_weight must be boolean",
)
@icontract.ensure(
    lambda result, splitter_supports_sample_weight: _bool(result)
    and result == splitter_supports_sample_weight,
    "splitter sample-weight forwarding must match sklearn routing logic",
)
def cd_cv_forward_splitter_sample_weight(splitter_supports_sample_weight: bool) -> bool:
    """Return whether LinearModelCV.fit forwards sample_weight into splitter params."""
    return splitter_supports_sample_weight


@register_atom(witness_cd_cv_drop_estimator_sample_weight)
@icontract.require(
    lambda sample_weight_is_not_none: _bool(sample_weight_is_not_none),
    "sample_weight_is_not_none must be boolean",
)
@icontract.require(
    lambda estimator_supports_sample_weight: _bool(estimator_supports_sample_weight),
    "estimator_supports_sample_weight must be boolean",
)
@icontract.ensure(
    lambda result, sample_weight_is_not_none, estimator_supports_sample_weight: _bool(result)
    and result == (sample_weight_is_not_none and (not estimator_supports_sample_weight)),
    "estimator sample-weight drop must match sklearn routing logic",
)
def cd_cv_drop_estimator_sample_weight(
    sample_weight_is_not_none: bool, estimator_supports_sample_weight: bool
) -> bool:
    """Return whether LinearModelCV.fit drops sample_weight for the estimator refit path."""
    return sample_weight_is_not_none and (not estimator_supports_sample_weight)


@register_atom(witness_cd_cv_default_routed_params_required)
@icontract.require(lambda routing_enabled: _bool(routing_enabled), "routing_enabled must be boolean")
@icontract.ensure(
    lambda result, routing_enabled: _bool(result) and result == (not routing_enabled),
    "default routed-params fallback must match the non-routing branch",
)
def cd_cv_default_routed_params_required(routing_enabled: bool) -> bool:
    """Return whether LinearModelCV.fit builds the default empty routed-params fallback."""
    return not routing_enabled
