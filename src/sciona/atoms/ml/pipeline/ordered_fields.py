"""Explicit field selection for graph inputs, independent of application schema."""
from sciona.ghost.registry import register_atom


def witness_select_values(*args, **kwargs):
    raise ValueError('Concrete mapping and explicit ordered field names required')


@register_atom(witness_select_values)
def select_values(mapping: dict, fields: list[str]) -> list:
    """Return borrowed mapping values in an explicit caller-selected order.

    Require a plain dict with nonblank string keys and a nonempty plain list or
    tuple of unique nonblank string names, all present in the dict. Extra mapping
    fields are permitted and ignored. Preserve exact names without trimming.
    Return a new outer list; values retain identity, dtype, shape and aliasing.
    Do not mutate inputs or copy, convert, inspect or validate values. Caller
    establishes field meanings, value contracts, alignment and units. Concrete
    execution only; this operation neither infers a schema nor runs models.
    """
    if type(mapping) is not dict or any(type(key) is not str or not key.strip() for key in mapping):
        raise ValueError('Plain dict with nonblank string keys required')
    if (type(fields) not in (list, tuple) or not fields
            or any(type(name) is not str or not name.strip() for name in fields)
            or len(set(fields)) != len(fields)):
        raise ValueError('Nonempty ordered unique field names required')
    if any(name not in mapping for name in fields):
        raise ValueError('Every selected field must exist')
    return [mapping[name] for name in fields]
