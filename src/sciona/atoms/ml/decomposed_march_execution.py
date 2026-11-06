"""March recipe adapters surrounding reusable binary sigmoid operations."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete inputs from the same invocation required')


@register_atom(concrete_only)
def features(prepared: object) -> dict:
    """Aggregate box scores, directed PageRank and chronological scaled Elo."""
    from sciona.decomposed_march_runtime import features as implementation
    return implementation(prepared)


@register_atom(concrete_only)
def pairings(featured: dict) -> dict:
    """Validate season chronology and build mirrored numeric populations."""
    from sciona.decomposed_march_runtime import pairings as implementation
    return implementation(featured)


@register_atom(concrete_only)
def fit_scores(arrays: dict) -> tuple[np.ndarray, np.ndarray, object, np.ndarray, np.ndarray, dict]:
    """Fit standardized binary logistic scores from arbitrary-width matrices."""
    from sciona.decomposed_march_runtime import fit_scores as implementation
    return implementation(arrays)


@register_atom(concrete_only)
def projection_inputs(forward: np.ndarray, reverse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expose orientation probabilities and coefficients for an ordered sum."""
    from sciona.decomposed_march_runtime import projection_inputs as implementation
    return implementation(forward, reverse)


@register_atom(concrete_only)
def result(total: np.ndarray, counts: dict) -> dict:
    """Scale the external sum and return probabilities with lifecycle counts."""
    from sciona.decomposed_march_runtime import result as implementation
    return implementation(total, counts)
