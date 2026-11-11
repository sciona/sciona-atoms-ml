"""Group-separated image/tabular recipe stages with reusable numeric decisions."""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete validated image/tabular state required')


@register_atom(concrete_only)
def prepare(payload: dict) -> object:
    """Validate and detach aligned images, metadata and group-separated populations."""
    from sciona._runtime.ml.image_tabular_complete_stages import prepare as run
    return run(payload)


@register_atom(concrete_only)
def fit(prepared: object) -> object:
    """Fit fold-local features and the retained binary neural ensemble."""
    from sciona._runtime.ml.image_tabular_complete_stages import fit as run
    return run(prepared)


@register_atom(concrete_only)
def calibrate(fitted: object) -> object:
    """Fit an empirical F1 threshold on separate calibration groups."""
    from sciona._runtime.ml.image_tabular_complete_stages import calibrate as run
    return run(fitted)


@register_atom(concrete_only)
def predict(calibrated: object) -> dict:
    """Apply the retained ensemble and decision threshold without refitting."""
    from sciona._runtime.ml.image_tabular_complete_stages import predict as run
    return run(calibrated)
