"""Deterministic sparse-encode precompute-dispatch helpers."""

from .atoms import (
    sparse_encode_dispatched_covariance,
    sparse_encode_dispatched_gram,
    sparse_encode_resolved_copy_cov,
)

__all__ = [
    "sparse_encode_dispatched_covariance",
    "sparse_encode_dispatched_gram",
    "sparse_encode_resolved_copy_cov",
]
