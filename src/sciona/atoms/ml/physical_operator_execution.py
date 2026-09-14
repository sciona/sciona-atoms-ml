"""Draft Community boundaries for generic periodic diffusion surrogates."""
from sciona.ghost.registry import register_atom


def witness_physical_operator_prepare(payload: dict) -> dict:return {'kind':'PhysicalOperator.Prepared'}


@register_atom(witness_physical_operator_prepare)
def physical_operator_prepare(payload: dict) -> object:
    """Validate SI periodic states, disjoint populations and training controls."""
    from sciona.physical_operator_contract import prepare
    return prepare(payload)


def witness_physical_operator_execute(prepared: object) -> dict:
    if prepared!={'kind':'PhysicalOperator.Prepared'}:raise ValueError('Prepared tabular input required')
    return {'kind':'PhysicalOperator.Result'}


@register_atom(witness_physical_operator_execute)
def physical_operator_execute(prepared: object) -> dict:
    """Train Fourier operator, restore validation checkpoint, project mass and smooth."""
    from sciona.physical_operator_contract import execute
    return execute(prepared)
