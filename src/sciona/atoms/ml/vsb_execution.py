"""Draft Community boundaries for the independent complete VSB pipeline."""
from sciona.ghost.registry import register_atom


def witness_vsb_prepare(payload: dict) -> dict:return {'kind':'VSB.Prepared'}


@register_atom(witness_vsb_prepare)
def vsb_prepare(payload: dict) -> object:
    """Validate private raw signal triples, labels and fold lifecycle."""
    from sciona.vsb_execution import prepare
    return prepare(payload)


def witness_vsb_execute(prepared: object) -> dict:
    if prepared!={'kind':'VSB.Prepared'}:raise ValueError('Prepared VSB input required')
    return {'kind':'VSB.Result'}


@register_atom(witness_vsb_execute)
def vsb_execute(prepared: object) -> dict:
    """Extract measurement features, fit full repeated ensemble and threshold signals."""
    from sciona.vsb_execution import execute
    return execute(prepared)
