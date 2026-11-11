"""Explicit keyed-query boundary between the two source inference branches."""
from sciona.ghost.registry import register_atom


def witness_queries(*args, **kwargs):
    raise ValueError('Concrete mapped query rows required')


@register_atom(witness_queries)
def split_keyed_queries(queries):
    """Return copied entity/time rows and their supplied keys in identical order.

    Require a nonempty plain DataFrame with exactly key, entity and time columns,
    unique column names, unique nonblank string keys, nonblank string entities
    and native int64 times. Preserve row order, index, exact strings and time
    dtype. Repeated entity/time pairs with different keys remain distinct rows.
    Return (entity/time DataFrame, key list); modifying either output does not
    modify input cell values. No sorting, key generation, parsing, fitting or
    numerical conversion occurs. Caller owns key meaning, time units, frozen
    query-scope agreement and consistent population mapping across branches.
    """
    import numpy as np
    import pandas as pd

    if (type(queries) is not pd.DataFrame or queries.empty
            or not queries.columns.is_unique
            or set(queries.columns) != {'key', 'entity', 'time'}):
        raise ValueError('Exact nonempty keyed-query DataFrame required')
    for field in ('key', 'entity'):
        if any(type(value) is not str or not value.strip() for value in queries[field]):
            raise ValueError('Nonblank string keys and entities required')
    if queries['key'].duplicated().any() or queries['time'].dtype != np.dtype('int64'):
        raise ValueError('Unique keys and native int64 query times required')
    return queries.loc[:, ['entity', 'time']].copy(deep=True), queries['key'].tolist()
