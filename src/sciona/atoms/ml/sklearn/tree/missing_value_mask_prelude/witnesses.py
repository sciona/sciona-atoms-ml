"""Ghost witnesses for sklearn tree missing-value-mask prelude atoms."""

from __future__ import annotations


def witness_tree_missing_values_estimator_name(estimator_name: str | None, class_name: str) -> str:
    """Describe estimator_name defaulting in _compute_missing_values_in_feature_mask."""
    return estimator_name or class_name


def witness_tree_missing_values_common_kwargs(estimator_name: str) -> dict[str, str]:
    """Describe common_kwargs construction in _compute_missing_values_in_feature_mask."""
    return {"estimator_name": estimator_name, "input_name": "X"}


def witness_tree_missing_values_overall_sum_requires_elementwise_check(overall_sum: float) -> bool:
    """Describe the non-finite overall_sum guard."""
    return not bool(__import__("numpy").isfinite(overall_sum))


def witness_tree_missing_values_overall_sum_has_missing(overall_sum: float) -> bool:
    """Describe the nan overall_sum missing-values branch."""
    return bool(__import__("numpy").isnan(overall_sum))

