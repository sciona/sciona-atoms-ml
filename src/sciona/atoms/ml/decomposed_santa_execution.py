"""Complete replay-trained Santa recipe with independently bound numeric blending."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete inputs from the same invocation required')


@register_atom(concrete_only)
def fit(prepared: object) -> object:
    """Generate all sampled replay rows and fit both held-out-selected models."""
    from sciona._runtime.ml.decomposed_santa_runtime import fit as implementation
    return implementation(prepared)


@register_atom(concrete_only)
def predictions(prepared: object, models: object) -> dict:
    """Predict both feature families with each model's selected iteration."""
    from sciona._runtime.ml.decomposed_santa_runtime import predictions as implementation
    return implementation(prepared, models)


@register_atom(concrete_only)
def blend_inputs(prepared: object, predicted: dict) -> tuple[np.ndarray, np.ndarray]:
    """Invert the exponential target and expose ordered recipe coefficients."""
    from sciona._runtime.ml.decomposed_santa_runtime import blend_inputs as implementation
    return implementation(prepared, predicted)


@register_atom(concrete_only)
def result(prepared: object, models: object, initial: np.ndarray) -> dict:
    """Apply shared decay, seeded ties and nonpositive fallback; retain fit evidence."""
    from sciona._runtime.ml.decomposed_santa_runtime import result as implementation
    return implementation(prepared, models, initial)
