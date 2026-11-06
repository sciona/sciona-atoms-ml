"""Complete M5 stages around a domain-independent numeric reduction."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete inputs from the same invocation required')


@register_atom(concrete_only)
def features(prepared: object) -> dict:
    """Construct the full prepared feature grid from explicit caller roles."""
    from sciona.decomposed_m5_runtime import features as implementation
    return implementation(prepared)


@register_atom(concrete_only)
def fit(prepared: object, transformed: dict) -> dict:
    """Train all pools in all six families with the full learner schedules."""
    from sciona.decomposed_m5_runtime import fit as implementation
    return implementation(prepared, transformed)


@register_atom(concrete_only)
def forecast(transformed: dict, fitted: dict) -> dict:
    """Forecast all families for 28 days with separate recursive feedback."""
    from sciona.decomposed_m5_runtime import forecast as implementation
    return implementation(transformed, fitted)


@register_atom(concrete_only)
def blend_inputs(forecasted: dict) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the six family matrices and expose unit coefficients."""
    from sciona.decomposed_m5_runtime import blend_inputs as implementation
    return implementation(forecasted)


@register_atom(concrete_only)
def result(prepared: object, forecasted: dict, total: np.ndarray) -> dict:
    """Divide the ordered sum by six and package the complete forecast."""
    from sciona.decomposed_m5_runtime import result as implementation
    return implementation(prepared, forecasted, total)
