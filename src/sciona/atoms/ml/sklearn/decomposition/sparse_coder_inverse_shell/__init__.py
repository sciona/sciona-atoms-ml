"""SparseCoder fit and inverse-transform shell helpers."""

from .atoms import (
    sparse_coder_fit_require_matching_features,
    sparse_coding_expected_code_width,
    sparse_coding_merge_split_sign,
    sparse_coder_inverse_transform,
)

__all__ = [
    "sparse_coder_fit_require_matching_features",
    "sparse_coding_expected_code_width",
    "sparse_coding_merge_split_sign",
    "sparse_coder_inverse_transform",
]
