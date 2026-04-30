"""Pure tabular feature-engineering helpers for gradient-boosting pipelines."""

from .atoms import (
    aggregate_child_table,
    extract_pseudo_labels,
    frequency_encode,
    frequency_encode_fit,
    group_aggregate,
    log_cosh_gradient,
    missing_indicator_and_impute,
    null_importance_p_values,
    pairwise_products,
    pairwise_ratios,
    rank_transform,
    rolling_statistics,
    target_encode,
    temporal_difference,
    time_decay_aggregate,
    tweedie_gradient,
)

__all__ = [
    "aggregate_child_table",
    "extract_pseudo_labels",
    "frequency_encode",
    "frequency_encode_fit",
    "group_aggregate",
    "log_cosh_gradient",
    "missing_indicator_and_impute",
    "null_importance_p_values",
    "pairwise_products",
    "pairwise_ratios",
    "rank_transform",
    "rolling_statistics",
    "target_encode",
    "temporal_difference",
    "time_decay_aggregate",
    "tweedie_gradient",
]

