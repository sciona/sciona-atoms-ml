"""Sklearn GLM tags/loss callback atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_glm_base_default_loss_name,
    witness_glm_tags_exception_fallback,
    witness_glm_tags_loss_callback_result,
    witness_glm_tags_positive_only_from_negative_range,
    witness_glm_tags_return,
    witness_glm_tags_sparse_input_value,
    witness_glm_tags_super_result,
)

_GLM_TAG_FALLBACK_ERRORS = {"ValueError", "AttributeError", "TypeError"}
_BASE_DEFAULT_LOSS_NAME = "HalfSquaredError"


@register_atom(witness_glm_tags_super_result)
@icontract.require(lambda tags_from_super: tags_from_super is not None, "tags_from_super must be provided")
@icontract.ensure(
    lambda result, tags_from_super: result is tags_from_super,
    "GLM super().__sklearn_tags__ result must preserve tags object identity",
)
def glm_tags_super_result(tags_from_super: object) -> object:
    """Return the tags object produced by GLM super().__sklearn_tags__."""
    return tags_from_super


@register_atom(witness_glm_tags_sparse_input_value)
@icontract.require(lambda tags: tags is not None, "tags must be provided")
@icontract.ensure(lambda result: result is True, "GLM input_tags.sparse must be set to True")
def glm_tags_sparse_input_value(tags: object) -> bool:
    """Return the fixed sparse-input tag value assigned by GLM tags."""
    return True


@register_atom(witness_glm_tags_loss_callback_result)
@icontract.require(lambda base_loss: base_loss is not None, "base_loss must be provided")
@icontract.ensure(
    lambda result, base_loss: result is base_loss,
    "GLM _get_loss callback result must preserve loss object identity",
)
def glm_tags_loss_callback_result(base_loss: object) -> object:
    """Return the loss object produced by the GLM _get_loss callback."""
    return base_loss


@register_atom(witness_glm_tags_positive_only_from_negative_range)
@icontract.require(
    lambda in_negative_range: isinstance(in_negative_range, bool),
    "loss range probe result must be a bool",
)
@icontract.ensure(
    lambda result, in_negative_range: result is (not in_negative_range),
    "positive_only must be the negation of in_y_true_range(-1.0)",
)
def glm_tags_positive_only_from_negative_range(in_negative_range: bool) -> bool:
    """Return the positive-only tag value derived from loss range probing."""
    return not in_negative_range


@register_atom(witness_glm_tags_exception_fallback)
@icontract.require(lambda tags: tags is not None, "tags must be provided")
@icontract.require(
    lambda error_type: error_type in _GLM_TAG_FALLBACK_ERRORS,
    "error_type must be one of the GLM tag fallback exceptions",
)
@icontract.ensure(
    lambda result, tags: result is tags,
    "GLM tag fallback must preserve the current tags object",
)
def glm_tags_exception_fallback(tags: object, error_type: str) -> object:
    """Return unchanged tags when GLM loss probing hits a fallback exception."""
    return tags


@register_atom(witness_glm_tags_return)
@icontract.require(lambda tags: tags is not None, "tags must be provided")
@icontract.ensure(
    lambda result, tags: result is tags,
    "GLM __sklearn_tags__ must return the mutated tags object",
)
def glm_tags_return(tags: object) -> object:
    """Return the final tags object from GLM __sklearn_tags__."""
    return tags


@register_atom(witness_glm_base_default_loss_name)
@icontract.require(
    lambda loss_name: loss_name == _BASE_DEFAULT_LOSS_NAME,
    "base GLM default loss must be HalfSquaredError",
)
@icontract.ensure(lambda result: result == _BASE_DEFAULT_LOSS_NAME, "default loss name must be stable")
def glm_base_default_loss_name(loss_name: str) -> str:
    """Return the base GLM default _get_loss class name."""
    return loss_name
