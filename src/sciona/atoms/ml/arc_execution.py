"""Community candidate boundaries for the full pinned ARC winning solver."""
from sciona.ghost.registry import register_atom


def witness_arc_prepare(payload: dict) -> dict:
    return {"kind": "ARC.Prepared"}


@register_atom(witness_arc_prepare)
def arc_prepare(payload: dict) -> object:
    """Validate private versioned grid tasks and explicit search budgets."""
    from sciona.arc_runtime import prepare
    return prepare(payload)


def witness_arc_execute(prepared: object) -> dict:
    if prepared != {"kind": "ARC.Prepared"}:
        raise ValueError("Prepared ARC input required")
    return {"kind": "ARC.Result"}


@register_atom(witness_arc_execute)
def arc_execute(prepared: object) -> dict:
    """Run four adaptive source phases and return ranked predictions/statuses."""
    from sciona.arc_runtime import execute
    return execute(prepared)
