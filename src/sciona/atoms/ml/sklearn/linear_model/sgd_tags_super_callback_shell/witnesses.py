"""Ghost witnesses for sklearn SGD tags super-callback atoms."""

from __future__ import annotations


def witness_sgd_tags_super_result(tags_from_super: object) -> object:
    """Describe the tags object returned by SGD super().__sklearn_tags__()."""
    return tags_from_super


def witness_sgd_tags_sparse_input_value(tags: object) -> bool:
    """Describe the sparse-input tag value assigned by SGD tags."""
    return tags is not None


def witness_sgd_tags_return(tags: object) -> object:
    """Describe the final tags object returned by SGD __sklearn_tags__."""
    return tags
