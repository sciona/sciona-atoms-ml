"""Ghost witnesses for sklearn GLM tags/loss callback atoms."""

from __future__ import annotations


def witness_glm_tags_super_result(tags_from_super: object) -> object:
    """Describe the tags object returned by GLM super().__sklearn_tags__()."""
    return tags_from_super


def witness_glm_tags_sparse_input_value(tags: object) -> bool:
    """Describe the sparse-input tag value assigned by GLM tags."""
    return tags is not None


def witness_glm_tags_loss_callback_result(base_loss: object) -> object:
    """Describe the BaseLoss object returned by GLM _get_loss()."""
    return base_loss


def witness_glm_tags_positive_only_from_negative_range(in_negative_range: bool) -> bool:
    """Describe positive-only tag derivation from loss range membership."""
    return not in_negative_range


def witness_glm_tags_exception_fallback(tags: object, error_type: str) -> object:
    """Describe GLM tag preservation when loss probing raises."""
    return tags


def witness_glm_tags_return(tags: object) -> object:
    """Describe the final tags object returned by GLM __sklearn_tags__."""
    return tags


def witness_glm_base_default_loss_name(loss_name: str) -> str:
    """Describe the base GLM default _get_loss class name."""
    return loss_name
