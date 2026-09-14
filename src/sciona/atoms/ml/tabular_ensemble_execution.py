"""Draft Community boundaries for generic binary tabular stacking."""
from sciona.ghost.registry import register_atom


def witness_tabular_ensemble_prepare(payload: dict) -> dict:return {'kind':'TabularEnsemble.Prepared'}


@register_atom(witness_tabular_ensemble_prepare)
def tabular_ensemble_prepare(payload: dict) -> object:
    """Validate disjoint tabular populations and explicit training controls."""
    from sciona.tabular_ensemble_contract import prepare
    return prepare(payload)


def witness_tabular_ensemble_execute(prepared: object) -> dict:
    if prepared!={'kind':'TabularEnsemble.Prepared'}:raise ValueError('Prepared tabular input required')
    return {'kind':'TabularEnsemble.Result'}


@register_atom(witness_tabular_ensemble_execute)
def tabular_ensemble_execute(prepared: object) -> dict:
    """Fit fold-local base models, OOF stacker, held-out sigmoid and predict."""
    from sciona.tabular_ensemble_contract import execute
    return execute(prepared)
