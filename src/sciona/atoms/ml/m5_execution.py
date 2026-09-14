"""Community execution boundaries for the independent M5 six-family pipeline."""
from sciona.ghost.registry import register_atom


def witness_m5_prepare(payload: dict) -> dict:return {'kind':'M5.Prepared'}


@register_atom(witness_m5_prepare)
def m5_prepare(payload: dict) -> object:
    """Validate private histories, calendar and price roles for full forecasting."""
    from sciona.m5_execution import prepare
    return prepare(payload)


def witness_m5_execute(prepared: object) -> dict:
    if prepared!={'kind':'M5.Prepared'}:raise ValueError('Prepared M5 input required')
    return {'kind':'M5.Result'}


@register_atom(witness_m5_execute)
def m5_execute(prepared: object) -> dict:
    """Train all six pooled families and produce the full 28-day mean forecast."""
    from sciona.m5_execution import execute
    return execute(prepared)
