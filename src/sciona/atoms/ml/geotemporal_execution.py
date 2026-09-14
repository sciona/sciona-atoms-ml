"""Draft Community boundaries for generic geotemporal regression."""
from sciona.ghost.registry import register_atom


def witness_geotemporal_prepare(payload: dict) -> dict:return {'kind':'Geotemporal.Prepared'}


@register_atom(witness_geotemporal_prepare)
def geotemporal_prepare(payload: dict) -> object:
    """Validate planar metric reference, time blocks and future queries."""
    from sciona.geotemporal_contract import prepare
    return prepare(payload)


def witness_geotemporal_execute(prepared: object) -> dict:
    if prepared!={'kind':'Geotemporal.Prepared'}:raise ValueError('Prepared tabular input required')
    return {'kind':'Geotemporal.Result'}


@register_atom(witness_geotemporal_execute)
def geotemporal_execute(prepared: object) -> dict:
    """Validate forward folds, refit tree fusion and smooth future predictions."""
    from sciona.geotemporal_contract import execute
    return execute(prepared)
