"""Helpers for sklearn multiclass output-code fit bookkeeping."""

from .atoms import (
    output_code_fit_estimator_count,
    output_code_fit_feature_names_in,
    output_code_fit_n_features_in,
    output_code_fit_require_nonempty_classes,
)

__all__ = [
    "output_code_fit_estimator_count",
    "output_code_fit_feature_names_in",
    "output_code_fit_n_features_in",
    "output_code_fit_require_nonempty_classes",
]
