"""Community candidate boundaries for corrected Connectomics inference."""
from sciona.ghost.registry import register_atom


def witness_connectomics_prepare(payload: dict) -> dict:
    return {'kind': 'Connectomics.Prepared'}


@register_atom(witness_connectomics_prepare)
def connectomics_prepare(payload: dict) -> object:
    """Validate and copy private nonnegative time-by-node signals."""
    from sciona.connectomics_runtime import prepare
    return prepare(payload)


def witness_connectomics_execute(prepared: object) -> dict:
    if prepared != {'kind': 'Connectomics.Prepared'}:
        raise ValueError('Prepared Connectomics input required')
    return {'kind': 'Connectomics.Result'}


@register_atom(witness_connectomics_execute)
def connectomics_execute(prepared: object) -> dict:
    """Run all corrected grid fits and optional source precedence blend."""
    from sciona.connectomics_runtime import execute
    return execute(prepared)
