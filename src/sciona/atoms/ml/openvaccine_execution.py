"""Draft Community boundaries for reconstructed OpenVaccine execution."""
from sciona.ghost.registry import register_atom


def witness_openvaccine_prepare(payload: dict) -> dict:
    return {'kind':'OpenVaccine.Prepared'}


@register_atom(witness_openvaccine_prepare)
def openvaccine_prepare(payload: dict) -> object:
    """Validate private populations, explicit member splits and round recipes."""
    from sciona.openvaccine_contract import prepare
    return prepare(payload)


def witness_openvaccine_execute(prepared: object) -> dict:
    if prepared!={'kind':'OpenVaccine.Prepared'}:raise ValueError('Prepared OpenVaccine input required')
    return {'kind':'OpenVaccine.Result'}


@register_atom(witness_openvaccine_execute)
def openvaccine_execute(prepared: object) -> dict:
    """Fold sequences, train/refine all members and predict from saved states."""
    from sciona.openvaccine_contract import execute
    return execute(prepared)
