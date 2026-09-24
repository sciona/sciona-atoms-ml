"""Lazy, hash-bound prediction graph control operations."""
from sciona.ghost.registry import register_atom
from sciona.guarded_prediction_graphs import guarded_prediction


def witness_materialized(*args: object, **kwargs: object) -> object:
    raise ValueError('Materialized branch execution required; static propagation is unsupported')


@register_atom(witness=witness_materialized)
def emit(value: object, receive: object) -> object:
    """Deliver a terminal value to the invoking in-memory controller."""
    if not callable(receive):
        raise ValueError('Callable controller receiver required')
    receive(value)
    return value


@register_atom(witness=witness_materialized)
async def predict(primary: dict, baseline: dict, inputs: dict, count: int, constant: float) -> tuple:
    """Run primary/baseline graphs sequentially, then an explicit constant fallback."""
    return await guarded_prediction(primary, baseline, inputs, count, constant)
