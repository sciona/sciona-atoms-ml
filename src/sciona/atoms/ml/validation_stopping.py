"""Reusable validation checkpoint policy, independent of estimator and metric."""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete stopping policy and finite metrics required')


@register_atom(concrete_only)
def initialize(patience: int, min_delta: float, mode: str) -> object:
    """Create immutable policy; strict improvements reset positive patience.

    mode is min or max. min_delta is an absolute nonnegative threshold. Ties
    retain the earlier checkpoint. No metric, checkpoint storage or training
    behavior is inferred.
    """
    from sciona._runtime.ml.validation_stopping import ValidationStopping
    return ValidationStopping(patience, min_delta, mode)


@register_atom(concrete_only)
def observe(state: object, value: float) -> object:
    """Return fresh state after one finite metric; reject observations after stop."""
    from sciona._runtime.ml.validation_stopping import ValidationStopping
    if type(state) is not ValidationStopping:
        raise ValueError('Concrete validation stopping state required')
    return state.observe(value)
