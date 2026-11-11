"""Complete Porto recipe stages around a domain-independent numeric sum."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete inputs from the same invocation required')


@register_atom(concrete_only)
def features(prepared: object) -> dict:
    """Prepare the full joint feature populations without using labels."""
    from sciona._runtime.ml.decomposed_porto_runtime import features as implementation
    return implementation(prepared)


@register_atom(concrete_only)
def fit(prepared: object, transformed: dict) -> dict:
    """Fit five full DAE/classifier branches and one prepared-feature tree."""
    from sciona._runtime.ml.decomposed_porto_runtime import fit as implementation
    return implementation(prepared, transformed)


@register_atom(concrete_only)
def blend_inputs(fitted: dict) -> tuple[np.ndarray, np.ndarray]:
    """Validate all probabilities and expose the five ordered neural rows."""
    from sciona._runtime.ml.decomposed_porto_runtime import blend_inputs as implementation
    return implementation(fitted)


@register_atom(concrete_only)
def result(prepared: object, fitted: dict, neural_sum: np.ndarray) -> dict:
    """Add the tree after neural reduction and retain complete result metadata."""
    from sciona._runtime.ml.decomposed_porto_runtime import result as implementation
    return implementation(prepared, fitted, neural_sum)
