"""Bounded temporal sparse recipe with explicit validation checkpoint selection."""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete private stream configuration required')


@register_atom(concrete_only)
def prepare(payload: dict) -> object:
    """Inspect bounded streams and validate temporal boundaries and selection controls."""
    from sciona._runtime.ml.temporal_sparse_selected_execution import prepare as run
    return run(payload)


@register_atom(concrete_only)
def execute(prepared: object) -> dict:
    """Replay verified snapshots, select sparse ensemble checkpoint, predict queries.

    Features, encoding, fitting and prediction remain fused for bounded streaming
    and deterministic snapshot lifetime. Validation scores select epochs and do
    not establish independent accuracy. Source descriptors remain private.
    """
    from sciona._runtime.ml.temporal_sparse_selected_execution import execute as run
    return run(prepared)
