"""Explicit subject-separated signal lifecycle using reusable F1 decisions."""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete validated signal state required')


@register_atom(concrete_only)
def prepare(payload: dict) -> object:
    """Validate finite aligned signals and disjoint subject populations."""
    from sciona._runtime.ml.biosignal_complete_stages import prepare as run
    return run(payload)


@register_atom(concrete_only)
def fit(prepared: object) -> object:
    """Fit training-only scaling and the dual waveform/spectrogram model."""
    from sciona._runtime.ml.biosignal_complete_stages import fit as run
    return run(prepared)


@register_atom(concrete_only)
def calibrate(fitted: object) -> object:
    """Fit a recording-level empirical F1 decision threshold on separate calibration rows."""
    from sciona._runtime.ml.biosignal_complete_stages import calibrate as run
    return run(fitted)


@register_atom(concrete_only)
def predict(calibrated: object) -> dict:
    """Predict query recordings and apply the fitted threshold without refitting."""
    from sciona._runtime.ml.biosignal_complete_stages import predict as run
    return run(calibrated)
