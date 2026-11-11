"""Source recipe wiring around independently reusable numerical filtering."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete prepared signals and filtered arrays required')


@register_atom(concrete_only)
def filter_configuration(prepared: object) -> tuple:
    """Expose source-specific taps and time axis for four generic filter calls."""
    from sciona._runtime.ml.decomposed_connectomics_runtime import filter_configuration as configure
    return configure(prepared)


@register_atom(concrete_only)
def execute_prefiltered(prepared: object, f1: np.ndarray, f2: np.ndarray,
                        f3: np.ndarray, f4: np.ndarray) -> dict:
    """Run the full corrected source sweep using explicitly supplied filters."""
    from sciona._runtime.ml.decomposed_connectomics_runtime import execute_prefiltered as execute
    return execute(prepared, f1, f2, f3, f4)
