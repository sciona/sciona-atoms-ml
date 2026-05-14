"""Deterministic sklearn SGD tags super-callback atoms."""

from .atoms import (
    sgd_tags_return,
    sgd_tags_sparse_input_value,
    sgd_tags_super_result,
)

__all__ = [
    "sgd_tags_super_result",
    "sgd_tags_sparse_input_value",
    "sgd_tags_return",
]
