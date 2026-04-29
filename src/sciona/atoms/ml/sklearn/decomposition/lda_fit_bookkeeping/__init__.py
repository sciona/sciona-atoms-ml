"""Deterministic LatentDirichletAllocation fit-bookkeeping helper atoms."""

from .atoms import (
    lda_batch_bounds,
    lda_check_nonnegative_dtype_names,
    lda_fit_converged,
    lda_fit_evaluate_iteration_due,
    lda_fit_use_online_batches,
    lda_partial_fit_first_call,
    lda_partial_fit_require_matching_feature_count,
)

__all__ = [
    "lda_batch_bounds",
    "lda_check_nonnegative_dtype_names",
    "lda_fit_converged",
    "lda_fit_evaluate_iteration_due",
    "lda_fit_use_online_batches",
    "lda_partial_fit_first_call",
    "lda_partial_fit_require_matching_feature_count",
]
