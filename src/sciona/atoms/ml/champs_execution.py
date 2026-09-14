"""Community candidate boundaries for the corrected full CHAMPS pipeline."""
from sciona.ghost.registry import register_atom


def witness_champs_prepare(payload: dict) -> dict:
    return {'kind':'CHAMPS.Prepared'}


@register_atom(witness_champs_prepare)
def champs_prepare(payload: dict) -> object:
    """Validate private raw geometry, coupling labels and all model schedules."""
    from sciona.champs_contract import prepare
    return prepare(payload)


def witness_champs_execute(prepared: object) -> dict:
    if prepared != {'kind':'CHAMPS.Prepared'}:
        raise ValueError('Prepared CHAMPS input required')
    return {'kind':'CHAMPS.Result'}


@register_atom(witness_champs_execute)
def champs_execute(prepared: object) -> dict:
    """Train all full graph transformers, reload complete states and blend predictions."""
    from sciona.champs_runtime import execute
    return execute(prepared)
