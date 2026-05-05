"""Ghost witnesses for sklearn tree missing-value-mask output atoms."""

from __future__ import annotations


def witness_tree_missing_values_none_result(mask_supported: bool, overall_sum_has_missing: bool) -> None:
    """Describe the None-return branches of _compute_missing_values_in_feature_mask."""
    del mask_supported
    del overall_sum_has_missing
    return None


def witness_tree_missing_values_mask_required(mask_supported: bool, overall_sum_has_missing: bool) -> bool:
    """Describe the branch predicate for computing the missing-value mask."""
    return mask_supported and overall_sum_has_missing


def witness_tree_missing_values_mask_result(missing_values_in_feature_mask: tuple[bool, ...]) -> tuple[bool, ...]:
    """Describe the final missing-value mask returned by _compute_missing_values_in_feature_mask."""
    return missing_values_in_feature_mask

