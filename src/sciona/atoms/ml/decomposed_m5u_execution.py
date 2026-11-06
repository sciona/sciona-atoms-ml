"""Complete uncertainty stages surrounding a domain-independent numeric sum."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete inputs from the same invocation required')


@register_atom(concrete_only)
def fit_forecast(prepared: object) -> dict:
    """Train all levels and perform full repeated quantile inference."""
    from sciona.decomposed_m5u_runtime import fit_forecast as implementation
    return implementation(prepared)


@register_atom(concrete_only)
def restoration_inputs(forecasted: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """Expose native-unit state, ordered median totals and unit coefficients."""
    from sciona.decomposed_m5u_runtime import restoration_inputs as implementation
    return implementation(forecasted)


@register_atom(concrete_only)
def restore(state: dict, total: np.ndarray) -> np.ndarray:
    """Use the generic total to apply the complete level correction."""
    from sciona.decomposed_m5u_runtime import restore as implementation
    return implementation(state,total)


@register_atom(concrete_only)
def result(forecasted: dict, restored: np.ndarray) -> dict:
    """Package all restored quantiles and horizons with complete model reports."""
    from sciona.decomposed_m5u_runtime import result as implementation
    return implementation(forecasted,restored)
