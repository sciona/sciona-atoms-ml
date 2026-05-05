"""Partial-dependence categorical-dispatch shell atoms adapted from scikit-learn."""

from .atoms import (
    partial_dependence_categorical_array,
    partial_dependence_categorical_bool_branch,
    partial_dependence_categorical_index_or_name_branch,
)

__all__ = [
    "partial_dependence_categorical_array",
    "partial_dependence_categorical_bool_branch",
    "partial_dependence_categorical_index_or_name_branch",
]
