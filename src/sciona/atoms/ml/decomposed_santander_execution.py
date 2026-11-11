"""Complete Santander stages surrounding independently reusable numeric blending."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete inputs from the same invocation required')


@register_atom(concrete_only)
def initial(prepared: object) -> dict:
    """Train the complete initial neural cross-validation lifecycle."""
    from sciona._runtime.ml.decomposed_santander_runtime import initial as implementation
    return implementation(prepared)


@register_atom(concrete_only)
def selections(prepared: object, initial_models: dict) -> dict:
    """Select both full pseudo-label tails before branch training."""
    from sciona._runtime.ml.decomposed_santander_runtime import selections as implementation
    return implementation(prepared, initial_models)


@register_atom(concrete_only)
def branches(prepared: object, selected: dict) -> dict:
    """Regenerate features and refit all neural and tree folds sequentially."""
    from sciona._runtime.ml.decomposed_santander_runtime import branches as implementation
    return implementation(prepared, selected)


@register_atom(concrete_only)
def blend_inputs(fitted: dict) -> tuple[np.ndarray, np.ndarray]:
    """Expose average-tie ranks and the explicit 2.1:1 recipe coefficients."""
    from sciona._runtime.ml.decomposed_santander_runtime import blend_inputs as implementation
    return implementation(fitted)


@register_atom(concrete_only)
def result(prepared: object, initial_models: dict, selected: dict, fitted: dict, weighted: np.ndarray) -> dict:
    """Divide the ordered sum by 3.1 and retain full branch selection evidence."""
    from sciona._runtime.ml.decomposed_santander_runtime import result as implementation
    return implementation(prepared, initial_models, selected, fitted, weighted)
