"""Deterministic sparse-encode precomputed-solver bookkeeping helpers."""

from .atoms import (
    sparse_encode_lasso_alpha,
    sparse_encode_omp_norms_squared,
    sparse_encode_precomputed_output,
    sparse_encode_writable_init,
)

__all__ = [
    "sparse_encode_lasso_alpha",
    "sparse_encode_omp_norms_squared",
    "sparse_encode_precomputed_output",
    "sparse_encode_writable_init",
]
