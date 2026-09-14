"""Draft Community generic chronological matchup pipeline."""
from sciona.ghost.registry import register_atom


def witness_march_prepare(payload: dict) -> dict:return {'kind':'March.Prepared'}


@register_atom(witness_march_prepare)
def march_prepare(payload: dict) -> object:
    """Validate serialized private game populations and explicit controls."""
    from sciona.march_contract import prepare
    return prepare(payload)


def witness_march_execute(prepared: object) -> dict:
    if prepared!={'kind':'March.Prepared'}:raise ValueError('Prepared March input required')
    return {'kind':'March.Result'}


@register_atom(witness_march_execute)
def march_execute(prepared: object) -> dict:
    """Build season features, fit, calibrate and predict chronological matchups."""
    from sciona.march_contract import execute
    return execute(prepared)
