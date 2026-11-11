"""Concrete geotemporal stages with explicit coordinate, clock and population policy."""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete validated geotemporal state required')


@register_atom(concrete_only)
def prepare(payload: dict) -> object:
    """Validate explicit planar units, forward query times and recipe controls."""
    from sciona._runtime.ml.geotemporal_stages import prepare as run
    return run(payload)


@register_atom(concrete_only)
def features(prepared: object) -> object:
    """Build causal context/lag features with strictly fold-local target history."""
    from sciona._runtime.ml.geotemporal_stages import features as run
    return run(prepared)


@register_atom(concrete_only)
def fit(featured: object) -> object:
    """Fit expanding forward folds, evaluate held-out rows, and refit all training rows."""
    from sciona._runtime.ml.geotemporal_stages import fit as run
    return run(featured)


@register_atom(concrete_only)
def predict(fitted: object) -> object:
    """Predict query features from the full-training model without refitting."""
    from sciona._runtime.ml.geotemporal_stages import predict as run
    return run(fitted)


@register_atom(concrete_only)
def finish(predicted: object) -> dict:
    """Clip and blend co-timed query neighbors under explicit geographic policy."""
    from sciona._runtime.ml.geotemporal_stages import finish as run
    return run(predicted)
