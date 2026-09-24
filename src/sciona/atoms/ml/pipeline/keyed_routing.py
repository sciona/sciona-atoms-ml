"""Small deterministic routing operations for explicitly expanded population graphs.

Mappings are caller-owned runtime values. Operations copy the outer dictionary
and borrow values without mutation; no model execution or domain policy occurs.
"""
from sciona.ghost.registry import register_atom


def witness_materialized(*args: object, **kwargs: object) -> object:
    raise ValueError('Keyed routing requires materialized keys; static mapping propagation is unsupported')


def _mapping(mapping):
    if not isinstance(mapping, dict) or any(not isinstance(key, str) or not key for key in mapping):
        raise ValueError('Mapping with nonempty string keys required')


@register_atom(witness=witness_materialized)
def pop_first(mapping: dict) -> tuple:
    """Return first insertion-ordered key, borrowed value and copied remainder."""
    _mapping(mapping)
    if not mapping:
        raise ValueError('Graph has more branches than active populations')
    key=next(iter(mapping))
    remainder=dict(mapping)
    value=remainder.pop(key)
    return key,value,remainder


@register_atom(witness=witness_materialized)
def singleton(key: str, value: object) -> dict:
    """Start an output mapping without an implicit domain key or global state."""
    _mapping({key:value})
    return {key:value}


@register_atom(witness=witness_materialized)
def insert_unique(mapping: dict, key: str, value: object) -> dict:
    """Copy and insert exactly one new key; never overwrite an earlier result."""
    _mapping(mapping)
    _mapping({key:value})
    if key in mapping:
        raise ValueError('Duplicate output population')
    return dict(mapping, **{key:value})


@register_atom(witness=witness_materialized)
def require_exhausted(remaining: dict, result: dict) -> dict:
    """Reject a graph with fewer branches than supplied populations."""
    _mapping(remaining)
    _mapping(result)
    if remaining:
        raise ValueError('Graph has fewer branches than active populations')
    return result
