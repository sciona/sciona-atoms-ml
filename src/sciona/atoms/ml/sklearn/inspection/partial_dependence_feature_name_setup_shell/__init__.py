"""Partial-dependence feature-name setup shell atoms adapted from scikit-learn."""

from .atoms import (
    partial_dependence_default_feature_names,
    partial_dependence_duplicate_feature_names_guard_required,
    partial_dependence_duplicate_feature_names_message,
    partial_dependence_use_column_names_tolist,
    partial_dependence_use_feature_names_tolist,
)

__all__ = [
    "partial_dependence_use_column_names_tolist",
    "partial_dependence_default_feature_names",
    "partial_dependence_use_feature_names_tolist",
    "partial_dependence_duplicate_feature_names_guard_required",
    "partial_dependence_duplicate_feature_names_message",
]
