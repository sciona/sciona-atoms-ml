"""Draft Community generic causal anomaly pipeline boundaries."""
from sciona.ghost.registry import register_atom


def witness_nab_prepare(payload: dict) -> dict:return {'kind':'NAB.Prepared'}


@register_atom(witness_nab_prepare)
def nab_prepare(payload: dict) -> object:
    """Validate explicit private observations, detector controls and scoring windows."""
    from sciona.nab_contract import prepare
    return prepare(payload)


def witness_nab_execute(prepared: object) -> dict:
    if prepared!={'kind':'NAB.Prepared'}:raise ValueError('Prepared NAB input required')
    return {'kind':'NAB.Result'}


@register_atom(witness_nab_execute)
def nab_execute(prepared: object) -> dict:
    """Run the causal forest, rolling threshold and corrected benchmark scoring."""
    from sciona.nab_contract import execute
    return execute(prepared)
