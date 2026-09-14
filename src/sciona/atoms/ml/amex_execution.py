"""Draft Community boundaries for the independent full Amex pipeline."""
from sciona.ghost.registry import register_atom


def witness_amex_prepare(payload: dict) -> dict:return {'kind':'Amex.Prepared'}


@register_atom(witness_amex_prepare)
def amex_prepare(payload: dict) -> object:
    """Validate private sequences, explicit category roles and fold populations."""
    from sciona.amex_execution import prepare
    return prepare(payload)


def witness_amex_execute(prepared: object) -> dict:
    if prepared!={'kind':'Amex.Prepared'}:raise ValueError('Prepared Amex input required')
    return {'kind':'Amex.Result'}


@register_atom(witness_amex_execute)
def amex_execute(prepared: object) -> dict:
    """Execute full raw preprocessing and all 25 models with literal source blend."""
    from sciona.amex_execution import execute
    return execute(prepared)
