"""Partial-dependence auto-recursion support shell atoms adapted from scikit-learn."""

from .atoms import (
    partial_dependence_gradient_boosting_recursion_supported,
    partial_dependence_recursion_supported_estimator,
    partial_dependence_tree_recursion_supported,
)

__all__ = [
    "partial_dependence_gradient_boosting_recursion_supported",
    "partial_dependence_tree_recursion_supported",
    "partial_dependence_recursion_supported_estimator",
]
