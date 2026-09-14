"""Draft Community boundaries for generic binary text-pair matching."""
from sciona.ghost.registry import register_atom


def witness_text_pair_prepare(payload: dict) -> dict:return {'kind':'TextPair.Prepared'}


@register_atom(witness_text_pair_prepare)
def text_pair_prepare(payload: dict) -> object:
    """Validate disjoint text-pair populations and explicit training controls."""
    from sciona.text_pair_contract import prepare
    return prepare(payload)


def witness_text_pair_execute(prepared: object) -> dict:
    if prepared!={'kind':'TextPair.Prepared'}:raise ValueError('Prepared text-pair input required')
    return {'kind':'TextPair.Result'}


@register_atom(witness_text_pair_execute)
def text_pair_execute(prepared: object) -> dict:
    """Fit lexical/LSA/context ensemble, select held-out F1 threshold and predict."""
    from sciona.text_pair_contract import execute
    return execute(prepared)
