"""Draft Community boundaries for generic temporal sparse regression."""
from sciona.ghost.registry import register_atom


def witness_temporal_sparse_prepare(payload: dict) -> dict:return {'kind':'TemporalSparse.Prepared'}


@register_atom(witness_temporal_sparse_prepare)
def temporal_sparse_prepare(payload: dict) -> object:
    """Inspect bounded private streams and chronological split boundaries."""
    from sciona.temporal_sparse_execution import prepare
    return prepare(payload)


def witness_temporal_sparse_execute(prepared: object) -> dict:
    if prepared!={'kind':'TemporalSparse.Prepared'}:raise ValueError('Prepared temporal sparse input required')
    return {'kind':'TemporalSparse.Result'}


@register_atom(witness_temporal_sparse_execute)
def temporal_sparse_execute(prepared: object) -> dict:
    """Fit sparse online ensemble, evaluate frozen future stream and predict."""
    from sciona.temporal_sparse_execution import execute
    return execute(prepared)
