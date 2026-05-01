"""Deterministic SparseCoder API-shell helpers."""

from .atoms import (
    sparse_coder_fit_return_self,
    sparse_coder_n_features_out,
    sparse_coder_preserves_dtype_tags,
    sparse_coder_requires_fit_tag,
    sparse_coder_transform_dictionary,
)

__all__ = [
    "sparse_coder_fit_return_self",
    "sparse_coder_n_features_out",
    "sparse_coder_preserves_dtype_tags",
    "sparse_coder_requires_fit_tag",
    "sparse_coder_transform_dictionary",
]
