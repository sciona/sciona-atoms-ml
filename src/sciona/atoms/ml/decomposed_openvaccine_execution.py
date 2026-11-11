"""Complete feature-level reconstruction surrounding an external numeric sum."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete inputs from the same invocation required')


@register_atom(concrete_only)
def features(prepared: object) -> dict:
    """Fold the three raw populations and expose validated dense features."""
    from sciona._runtime.ml.decomposed_openvaccine_runtime import features as implementation
    return implementation(prepared)


@register_atom(concrete_only)
def fit_forecast(featured: dict) -> dict:
    """Run full pretraining, twenty models, all refinement rounds and inference."""
    from sciona._runtime.ml.decomposed_openvaccine_runtime import fit_forecast as implementation
    return implementation(featured)


@register_atom(concrete_only)
def blend_inputs(forecasted: dict) -> tuple[np.ndarray, np.ndarray]:
    """Expose ordered clipped predictions and unit sum coefficients."""
    from sciona._runtime.ml.decomposed_openvaccine_ensemble import blend_inputs as implementation
    return implementation(forecasted)


@register_atom(concrete_only)
def result(forecasted: dict, total: np.ndarray) -> dict:
    """Restore mean prediction shape and package teacher moments and counters."""
    from sciona._runtime.ml.decomposed_openvaccine_runtime import result as implementation
    return implementation(forecasted, total)
