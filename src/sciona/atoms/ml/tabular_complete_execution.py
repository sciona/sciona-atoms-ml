"""Draft concrete adapters for expanded binary mixed-table stacking.

Schema, feature applicability, supervised objectives and split choices remain
recipe policy. Numeric/text primitives remain independently reusable. Abstract
execution and arbitrary forged intermediate objects are not qualified.
"""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete validated recipe state required')


@register_atom(concrete_only)
def prepare(payload: dict) -> object:
    """Validate versioned feature policies and disjoint grouped populations."""
    from sciona.tabular_complete_contract import prepare as run
    return run(payload)


@register_atom(concrete_only)
def fit_stack(prepared: object) -> object:
    """Fit fold-local features, logistic/ExtraTrees OOF predictions and stacker."""
    from sciona.tabular_complete_stages import fit_stack as run
    return run(prepared)


@register_atom(concrete_only)
def calibrate(fitted: object) -> object:
    """Fit held-out sigmoid calibration without refitting base models or features."""
    from sciona.tabular_complete_stages import calibrate as run
    return run(fitted)


@register_atom(concrete_only)
def predict(calibrated: object) -> dict:
    """Predict ordered query probabilities and inclusive 0.5 binary decisions."""
    from sciona.tabular_complete_stages import predict as run
    return run(calibrated)
