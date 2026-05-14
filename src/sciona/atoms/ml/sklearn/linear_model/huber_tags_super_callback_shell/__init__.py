"""Deterministic sklearn Huber tags super-callback atoms."""

from .atoms import (
    huber_tags_return,
    huber_tags_sparse_input_value,
    huber_tags_super_result,
)

__all__ = [
    "huber_tags_super_result",
    "huber_tags_sparse_input_value",
    "huber_tags_return",
]
