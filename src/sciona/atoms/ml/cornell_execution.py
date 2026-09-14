"""Community candidate boundaries for corrected Cornell inference."""
from sciona.ghost.registry import register_atom


def witness_cornell_prepare(payload: dict) -> dict:
    return {'kind': 'Cornell.Prepared'}


@register_atom(witness_cornell_prepare)
def cornell_prepare(payload: dict) -> object:
    """Validate private audio, fold populations and complete lifecycle settings."""
    from sciona.cornell_contract import prepare
    return prepare(payload)


def witness_cornell_execute(prepared: object) -> dict:
    if prepared != {'kind': 'Cornell.Prepared'}:
        raise ValueError('Prepared Cornell input required')
    return {'kind': 'Cornell.Result'}


@register_atom(witness_cornell_execute)
def cornell_execute(prepared: object) -> dict:
    """Train all source members, reload selected checkpoints and vote on predictions."""
    from sciona.cornell_contract import execute
    return execute(prepared)
