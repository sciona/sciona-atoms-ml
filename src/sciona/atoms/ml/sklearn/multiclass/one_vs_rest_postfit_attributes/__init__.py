"""Helpers for sklearn multiclass one-vs-rest post-fit attributes."""

from .atoms import (
    one_vs_rest_fit_classes,
    one_vs_rest_fit_feature_names_in,
    one_vs_rest_fit_n_features_in,
)

__all__ = [
    "one_vs_rest_fit_classes",
    "one_vs_rest_fit_feature_names_in",
    "one_vs_rest_fit_n_features_in",
]
