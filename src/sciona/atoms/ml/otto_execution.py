"""Draft Community boundaries for the complete independent Otto CPU stack."""
from sciona.ghost.registry import register_atom


def witness_otto_prepare(payload: dict) -> dict:
    return {'kind':'Otto.Prepared'}


@register_atom(witness_otto_prepare)
def otto_prepare(payload: dict) -> object:
    """Validate private input populations and explicit execution controls."""
    from sciona.otto_execution import prepare
    return prepare(payload)


def witness_otto_execute(prepared: object) -> dict:
    if prepared!={'kind':'Otto.Prepared'}:
        raise ValueError('Prepared Otto input required')
    return {'kind':'Otto.Result'}


@register_atom(witness_otto_execute)
def otto_execute(prepared: object) -> dict:
    """Execute all first-level producers, meta selection/refits and final blend."""
    from sciona.otto_execution import execute
    return execute(prepared)
