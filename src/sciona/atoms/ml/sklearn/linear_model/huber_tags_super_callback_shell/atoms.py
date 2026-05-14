"""Sklearn Huber tags super-callback atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_huber_tags_return,
    witness_huber_tags_sparse_input_value,
    witness_huber_tags_super_result,
)


@register_atom(witness_huber_tags_super_result)
@icontract.require(lambda tags_from_super: tags_from_super is not None, "tags_from_super must be provided")
@icontract.ensure(
    lambda result, tags_from_super: result is tags_from_super,
    "HuberRegressor super().__sklearn_tags__ result must preserve tags object identity",
)
def huber_tags_super_result(tags_from_super: object) -> object:
    """Return the tags object produced by HuberRegressor super().__sklearn_tags__."""
    return tags_from_super


@register_atom(witness_huber_tags_sparse_input_value)
@icontract.require(lambda tags: tags is not None, "tags must be provided")
@icontract.ensure(lambda result: result is True, "HuberRegressor input_tags.sparse must be set to True")
def huber_tags_sparse_input_value(tags: object) -> bool:
    """Return the fixed sparse-input tag value assigned by HuberRegressor tags."""
    return True


@register_atom(witness_huber_tags_return)
@icontract.require(lambda tags: tags is not None, "tags must be provided")
@icontract.ensure(
    lambda result, tags: result is tags,
    "HuberRegressor __sklearn_tags__ must return the mutated tags object",
)
def huber_tags_return(tags: object) -> object:
    """Return the final tags object from HuberRegressor __sklearn_tags__."""
    return tags
