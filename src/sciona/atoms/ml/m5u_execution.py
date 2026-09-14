"""Community boundaries for the corrected direct-quantile uncertainty method."""
from sciona.ghost.registry import register_atom


def witness_m5u_prepare(payload: dict) -> dict:return {'kind':'M5U.Prepared'}


@register_atom(witness_m5u_prepare)
def m5u_prepare(payload: dict) -> object:
    """Validate private history, hierarchy and calendar for quantile forecasting."""
    from sciona.m5u_execution import prepare
    return prepare(payload)


def witness_m5u_execute(prepared: object) -> dict:
    if prepared!={'kind':'M5U.Prepared'}:raise ValueError('Prepared uncertainty input required')
    return {'kind':'M5U.Result'}


@register_atom(witness_m5u_execute)
def m5u_execute(prepared: object) -> dict:
    """Train all direct quantile levels and forecast with corrected normalization."""
    from sciona.m5u_execution import execute
    return execute(prepared)
