"""LatentDirichletAllocation variational-bound helpers."""

from .atoms import (
    lda_apply_subsampling_ratio,
    lda_approx_bound_from_expectations,
    lda_dirichlet_loglikelihood,
    lda_document_log_probability_bound,
)

__all__ = [
    "lda_dirichlet_loglikelihood",
    "lda_document_log_probability_bound",
    "lda_apply_subsampling_ratio",
    "lda_approx_bound_from_expectations",
]

