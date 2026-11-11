"""Concrete reusable numeric constraints, independent of physical recipe policy."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete finite numeric arrays required')


@register_atom(concrete_only)
def project_nonnegative_fixed_sum(values: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Project each finite numeric row onto nonnegativity and an explicit sum.

    totals has one finite nonnegative entry per row. Rows and columns must be
    nonempty. Return fresh float64 values; reject unrepresentable arithmetic.
    No physical units, probability normalization or target meaning are inferred.
    """
    from sciona._runtime.ml.constrained_array_ops import project_nonnegative_fixed_sum as run
    return run(values, totals)


@register_atom(concrete_only)
def smooth_periodic_rows(values: np.ndarray, weight: float) -> np.ndarray:
    """Apply a convex three-point circular stencil to each finite numeric row.

    Explicit weight lies in [0,.5]. Return fresh float64 values; signed signals
    are allowed. Preserve row sums to floating-point tolerance. Periodicity is
    a caller choice, not an inferred boundary condition or PDE solver claim.
    """
    from sciona._runtime.ml.constrained_array_ops import smooth_periodic_rows as run
    return run(values, weight)
