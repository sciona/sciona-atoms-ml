"""Port projection for the source frozen-feature result; no numerical conversion."""
from sciona.ghost.registry import register_atom


def witness_projection(*args,**kwargs):
    raise ValueError('Concrete frozen feature result required')


@register_atom(witness_projection)
def unpack_quantile_features(prepared):
    """Expose features, structured columns, observation mask and values in order.

    Require a plain mapping with exactly the four frozen-replay fields. Return
    borrowed values as a tuple without conversion or mutation. This operation
    only projects ports: upstream replay owns value validation and caller owns
    units, availability, alignment and downstream mutation of borrowed values.
    """
    names=('features','feature_columns','observed_mask','observed_values')
    if type(prepared) is not dict or set(prepared)!=set(names):
        raise ValueError('Exact frozen-replay output mapping required')
    return tuple(prepared[name] for name in names)
