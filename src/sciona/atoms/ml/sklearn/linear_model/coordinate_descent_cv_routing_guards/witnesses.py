"""Ghost witnesses for sklearn coordinate-descent CV routing-guard atoms."""

from __future__ import annotations


def witness_cd_cv_routing_enabled_branch(routing_enabled: object) -> object:
    """Describe the `_routing_enabled()` branch in LinearModelCV.fit."""
    return routing_enabled


def witness_cd_cv_sample_weight_support_guard_required(
    sample_weight_is_not_none: object,
    splitter_supports_sample_weight: object,
    estimator_supports_sample_weight: object,
) -> object:
    """Describe the unsupported sample-weight guard in LinearModelCV.fit."""
    return (
        sample_weight_is_not_none,
        splitter_supports_sample_weight,
        estimator_supports_sample_weight,
    )


def witness_cd_cv_sample_weight_support_message(guard_required: object) -> object:
    """Describe the unsupported sample-weight ValueError text in LinearModelCV.fit."""
    return guard_required


def witness_cd_cv_forward_splitter_sample_weight(splitter_supports_sample_weight: object) -> object:
    """Describe the splitter sample-weight forwarding branch in LinearModelCV.fit."""
    return splitter_supports_sample_weight


def witness_cd_cv_drop_estimator_sample_weight(
    sample_weight_is_not_none: object, estimator_supports_sample_weight: object
) -> object:
    """Describe the estimator sample-weight drop branch in LinearModelCV.fit."""
    return sample_weight_is_not_none, estimator_supports_sample_weight


def witness_cd_cv_default_routed_params_required(routing_enabled: object) -> object:
    """Describe the default routed-params fallback branch in LinearModelCV.fit."""
    return routing_enabled
