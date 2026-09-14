"""Draft Community boundaries for the independent Porto ensemble."""
from sciona.ghost.registry import register_atom


def witness_porto_prepare(payload: dict) -> dict:return {'kind':'Porto.Prepared'}


@register_atom(witness_porto_prepare)
def porto_prepare(payload: dict) -> object:
    """Validate private populations and explicit ensemble configuration."""
    from sciona.porto_execution import prepare
    return prepare(payload)


def witness_porto_execute(prepared: object) -> dict:
    if prepared!={'kind':'Porto.Prepared'}:raise ValueError('Prepared Porto input required')
    return {'kind':'Porto.Result'}


@register_atom(witness_porto_execute)
def porto_execute(prepared: object) -> dict:
    """Train five DAE neural branches and one tree, then average probabilities."""
    from sciona.porto_execution import execute
    return execute(prepared)
