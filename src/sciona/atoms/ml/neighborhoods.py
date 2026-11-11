"""Reusable numeric neighborhoods with caller-defined adjacency and meaning."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete finite values and a boolean neighborhood mask required')


@register_atom(concrete_only)
def blend_neighbor_means(values: np.ndarray, neighbors: np.ndarray, strength: float) -> np.ndarray:
    """Blend finite signed values with means selected by a square boolean mask.

    strength lies in [0,1]. Empty neighborhoods retain the original value;
    selected self-neighbors are allowed. Return a fresh float64 vector. No
    clipping, units, spatial/temporal adjacency or sum conservation is implied.
    """
    from sciona._runtime.ml.neighborhood_ops import blend_neighbor_means as run
    return run(values, neighbors, strength)
