"""Ghost witnesses for sklearn tree missing-value support atoms."""

from __future__ import annotations


def witness_tree_missing_values_x_is_sparse(x_is_sparse: bool) -> bool:
    """Describe whether the tree input is sparse."""
    return x_is_sparse


def witness_tree_missing_values_allow_nan_enabled(allow_nan_tag: bool) -> bool:
    """Describe whether tree tags permit NaN input."""
    return allow_nan_tag


def witness_tree_missing_values_monotonic_constraints_absent(monotonic_cst_is_none: bool) -> bool:
    """Describe whether monotonic constraints are absent."""
    return monotonic_cst_is_none


def witness_tree_missing_values_supported(
    *,
    x_is_sparse: bool,
    allow_nan_tag: bool,
    monotonic_cst_is_none: bool,
) -> bool:
    """Describe BaseDecisionTree._support_missing_values."""
    return (not x_is_sparse) and allow_nan_tag and monotonic_cst_is_none

