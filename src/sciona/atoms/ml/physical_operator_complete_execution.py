"""Draft concrete stages for explicit periodic scalar diffusion policy.

Units, split identities, model family and conservation policy are recipe choices.
Only validated concrete execution is qualified; no abstract numerical witness.
"""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete validated operator state required')


@register_atom(concrete_only)
def prepare(payload: dict) -> object:
    """Validate explicit units, periodic grid and separated populations."""
    from sciona.physical_operator_stages import prepare as run
    return run(payload)


@register_atom(concrete_only)
def fit(prepared: object) -> object:
    """Train Fourier/local blocks and restore the best validation checkpoint."""
    from sciona.physical_operator_stages import fit as run
    return run(prepared)


@register_atom(concrete_only)
def predict(fitted: object) -> object:
    """Predict ordered query rows using the frozen fitted operator."""
    from sciona.physical_operator_stages import predict as run
    return run(fitted)


@register_atom(concrete_only)
def project(predicted: object) -> object:
    """Project query rows onto their nonnegative initial-sum constraints."""
    from sciona.physical_operator_stages import project as run
    return run(predicted)


@register_atom(concrete_only)
def finish(projected: object) -> dict:
    """Smooth periodic rows under explicit policy and return finite results."""
    from sciona.physical_operator_stages import finish as run
    return run(projected)
