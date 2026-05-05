"""Sklearn tree missing-value-mask output atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_missing_values_mask_required,
    witness_tree_missing_values_mask_result,
    witness_tree_missing_values_none_result,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _mask(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) >= 1
        and all(isinstance(item, bool) for item in value)
    )


@register_atom(witness_tree_missing_values_none_result)
@icontract.require(lambda mask_supported: _bool(mask_supported), "mask_supported must be boolean")
@icontract.require(lambda overall_sum_has_missing: _bool(overall_sum_has_missing), "overall_sum_has_missing must be boolean")
@icontract.require(
    lambda mask_supported, overall_sum_has_missing: (not mask_supported) or (not overall_sum_has_missing),
    "None-result branch only applies when missing-value support is disabled or overall_sum shows no missing values",
)
@icontract.ensure(lambda result: result is None, "result must be None")
def tree_missing_values_none_result(
    *,
    mask_supported: bool,
    overall_sum_has_missing: bool,
) -> None:
    """Return sklearn's None branch for unsupported or missing-free inputs."""
    return None


@register_atom(witness_tree_missing_values_mask_required)
@icontract.require(lambda mask_supported: _bool(mask_supported), "mask_supported must be boolean")
@icontract.require(lambda overall_sum_has_missing: _bool(overall_sum_has_missing), "overall_sum_has_missing must be boolean")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def tree_missing_values_mask_required(
    *,
    mask_supported: bool,
    overall_sum_has_missing: bool,
) -> bool:
    """Return whether sklearn proceeds to compute _any_isnan_axis0(X)."""
    return mask_supported and overall_sum_has_missing


@register_atom(witness_tree_missing_values_mask_result)
@icontract.require(
    lambda missing_values_in_feature_mask: _mask(missing_values_in_feature_mask),
    "missing_values_in_feature_mask must be a nonempty boolean tuple",
)
@icontract.ensure(
    lambda result, missing_values_in_feature_mask: _mask(result) and result == missing_values_in_feature_mask,
    "mask result must preserve the supplied boolean tuple",
)
def tree_missing_values_mask_result(
    missing_values_in_feature_mask: tuple[bool, ...],
) -> tuple[bool, ...]:
    """Return sklearn's final missing-values-per-feature mask."""
    return missing_values_in_feature_mask

