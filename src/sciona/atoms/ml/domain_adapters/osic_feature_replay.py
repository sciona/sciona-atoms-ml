"""Explicit frozen preprocessing boundary for the source quantile recipe."""
from sciona.ghost.registry import register_atom


def witness_feature_replay(*args, **kwargs):
    raise ValueError('Concrete private preparation state and exact mapped queries required')


@register_atom(witness_feature_replay)
def transform_frozen_queries(state, queries, feature_coordinate, response_coordinate):
    """Transform the exact declared query batch using saved source feature state.

    Require private caller-owned state, mapped entity/time queries and matching
    nonblank feature and response coordinates. Preserve frozen category order,
    arithmetic dtypes, scaling ranges, observed baselines and query order.
    Reject a changed query population or order: this transductive state is not
    an unrestricted train-only scaler. Return a mapping with features, structured
    feature_columns, observed_mask and observed_values. Do not cast features to
    model input dtype, serialize column descriptors, restore models or reduce
    predictions. Those are separate graph operations.

    No fitting or extrema discovery occurs. Caller establishes state provenance,
    category meanings, coordinate units and observation availability. Structural
    state checks do not authenticate state or establish empirical transfer.
    Saved observations and query scope must remain private runtime material.
    """
    from sciona.osic_feature_state import transform_queries
    return transform_queries(state, queries, feature_coordinate=feature_coordinate,
                             response_coordinate=response_coordinate)
