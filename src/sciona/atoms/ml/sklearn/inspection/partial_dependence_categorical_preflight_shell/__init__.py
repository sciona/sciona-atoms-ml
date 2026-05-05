"""Partial-dependence categorical-preflight shell atoms adapted from scikit-learn."""

from .atoms import (
    partial_dependence_categorical_bool_size_guard_required,
    partial_dependence_categorical_bool_size_message,
    partial_dependence_categorical_dtype_message,
    partial_dependence_categorical_dtype_supported,
    partial_dependence_categorical_empty_guard_required,
    partial_dependence_categorical_empty_message,
)

__all__ = [
    "partial_dependence_categorical_empty_guard_required",
    "partial_dependence_categorical_empty_message",
    "partial_dependence_categorical_bool_size_guard_required",
    "partial_dependence_categorical_bool_size_message",
    "partial_dependence_categorical_dtype_supported",
    "partial_dependence_categorical_dtype_message",
]
