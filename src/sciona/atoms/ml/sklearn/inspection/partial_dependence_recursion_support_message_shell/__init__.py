"""Partial-dependence recursion-support message shell atoms adapted from scikit-learn."""

from .atoms import (
    partial_dependence_recursion_support_guard_required,
    partial_dependence_supported_recursion_classes,
    partial_dependence_unsupported_recursion_message,
)

__all__ = [
    "partial_dependence_recursion_support_guard_required",
    "partial_dependence_supported_recursion_classes",
    "partial_dependence_unsupported_recursion_message",
]
