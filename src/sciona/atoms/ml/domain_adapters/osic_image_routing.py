"""Explicit population orchestration; numerical computations stay injected."""
from sciona.ghost.registry import register_atom


def witness_image_routing(*args,**kwargs):
    raise ValueError('Concrete mapped populations and explicit computation bindings required')


@register_atom(witness_image_routing)
def route_and_predict(baselines, queries, planes, model, batch_size,
                      prepare_function, select_function, predict_function):
    """Route image entities sequentially and expose inputs for separate reduction.

    Require exact mapped baseline/query tables, unique identities and full plane
    coverage. Preserve baseline entity and query order. Use explicit source
    batch size 32. Invoke supplied preparation, ordered field selection and
    prediction once per entity; callables and model code remain caller-trusted.
    Validate finite scalar slice predictions and consistent float32/float64 dtype.
    Release local prepared-image references between entities; arbitrary callbacks
    can retain their own references, so this is not a global memory bound.

    Return slice-slope lists, baseline time/value/uncertainty vectors, query entity
    indices, query times, query keys and selected positions as an eight-item tuple.
    No trajectory reduction, model construction, fitting or branch blending is
    performed by this dispatcher. Caller owns units, mappings and availability.
    Bound compute threads and reserve the prediction callback's dataset worker.
    """
    from sciona._runtime.ml.osic_image_routing import route_and_predict as execute
    return execute(baselines,queries,planes,model,batch_size,
                   prepare_function,select_function,predict_function)
