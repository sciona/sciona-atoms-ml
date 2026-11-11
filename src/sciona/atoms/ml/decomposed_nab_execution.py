"""Full generic NAB recipe with a separately bound rolling numeric threshold."""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete inputs from the same invocation required')


@register_atom(concrete_only)
def forest_scores(prepared: object) -> dict:
    """Fit every causal trailing-window forest and retain every anomaly score."""
    from sciona._runtime.ml.decomposed_nab_runtime import forest_scores as implementation
    return implementation(prepared)


@register_atom(concrete_only)
def threshold_inputs(prepared: object, forest: dict) -> tuple[list, int, int, int, float]:
    """Remove only warmup slots and expose the recipe's three-sigma policy."""
    from sciona._runtime.ml.decomposed_nab_runtime import threshold_inputs as implementation
    p = implementation(prepared, forest)
    return p['values'], p['history'], p['minimum'], p['ddof'], p['multiplier']


@register_atom(concrete_only)
def detection(prepared: object, forest: dict, thresholds: list) -> dict:
    """Restore warmup slots and apply strict greater-than detection decisions.

    Inputs must be from one graph invocation; structural validation does not
    authenticate forest or threshold computation provenance.
    """
    from sciona._runtime.ml.decomposed_nab_runtime import detection as implementation
    return implementation(prepared, forest, thresholds)


@register_atom(concrete_only)
def result(prepared: object, detected: dict) -> dict:
    """Evaluate complete detections with corrected NAB scoring and normalization."""
    from sciona._runtime.ml.decomposed_nab_runtime import result as implementation
    return implementation(prepared, detected)
