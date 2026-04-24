"""Deterministic bagging fit-shell bookkeeping helpers."""

from .atoms import (
    bagging_additional_estimator_count,
    bagging_fit_require_bootstrap_for_oob,
    bagging_fit_require_no_warm_start_with_oob,
    bagging_resolve_max_features,
    bagging_resolve_max_samples,
)

__all__ = [
    "bagging_additional_estimator_count",
    "bagging_fit_require_bootstrap_for_oob",
    "bagging_fit_require_no_warm_start_with_oob",
    "bagging_resolve_max_features",
    "bagging_resolve_max_samples",
]
