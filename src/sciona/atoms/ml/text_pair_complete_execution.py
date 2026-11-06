"""Draft concrete recipe adapters for separate-field binary relevance.

Numerical primitives remain independently reusable. These adapters carry the
explicit recipe policy and execution-local fitted state; abstract execution
and arbitrary caller-constructed intermediate objects are not qualified.
"""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete validated recipe state required')


@register_atom(concrete_only)
def prepare_fields(payload: dict) -> object:
    """Validate and copy explicit policies and disjoint title/description rows."""
    from sciona.text_pair_complete_stages import prepare_fields as run
    return run(payload)


@register_atom(concrete_only)
def normalize_fields(prepared: object) -> object:
    """Normalize separate fields with caller policy and check population overlap."""
    from sciona.text_pair_complete_stages import normalize_fields as run
    return run(prepared)


@register_atom(concrete_only)
def fit_features(normalized: object) -> object:
    """Fit description vocabulary, projection, BM25 and context on training only."""
    from sciona.text_pair_complete_stages import fit_features as run
    return run(normalized)


@register_atom(concrete_only)
def fit_models(features: object) -> object:
    """Fit the explicit logistic/random-forest recipe using training labels only."""
    from sciona.text_pair_complete_stages import fit_models as run
    return run(features)


@register_atom(concrete_only)
def predict(models: object) -> object:
    """Fit isotonic mapping and F1 cutoff on calibration rows, then predict query rows."""
    from sciona.text_pair_complete_stages import predict as run
    return run(models)


@register_atom(concrete_only)
def format_rows(predictions: object) -> dict:
    """Emit finite ordered relevance rows and calibration selection metrics."""
    from sciona.text_pair_complete_stages import format_rows as run
    return run(predictions)
