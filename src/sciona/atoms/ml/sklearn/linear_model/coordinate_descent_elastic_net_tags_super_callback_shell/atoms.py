"""Sklearn ElasticNet tags super-callback atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_elastic_net_tags_return,
    witness_cd_elastic_net_tags_super_result,
)


@register_atom(witness_cd_elastic_net_tags_super_result)
@icontract.require(lambda tags_from_super: tags_from_super is not None, "tags_from_super must be provided")
@icontract.ensure(
    lambda result, tags_from_super: result is tags_from_super,
    "ElasticNet super().__sklearn_tags__ result must preserve tags object identity",
)
def cd_elastic_net_tags_super_result(tags_from_super: object) -> object:
    """Return the tags object produced by ElasticNet super().__sklearn_tags__."""
    return tags_from_super


@register_atom(witness_cd_elastic_net_tags_return)
@icontract.require(lambda tags: tags is not None, "tags must be provided")
@icontract.ensure(
    lambda result, tags: result is tags,
    "ElasticNet.__sklearn_tags__ must return the mutated tags object",
)
def cd_elastic_net_tags_return(tags: object) -> object:
    """Return the final tags object from ElasticNet.__sklearn_tags__."""
    return tags
