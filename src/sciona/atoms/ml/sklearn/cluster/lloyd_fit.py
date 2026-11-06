"""Reusable dense Lloyd clustering through the installed scikit-learn backend."""
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete finite points required for iterative clustering')


def _integer(value, minimum, maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError('Integer control outside supported bounds')
    return int(value)


@register_atom(concrete_only)
def fit_lloyd_clusters(points: np.ndarray, n_clusters: int, random_state: int = 0,
                       n_init: int = 1, max_iter: int = 300,
                       tolerance: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Return centers (K,D) and labels (N,) for dense Euclidean clustering.

    Requires a nonempty finite float32/float64 matrix. Preserves input precision,
    copies storage, uses k-means++ initialization and Lloyd updates, and bounds
    native threads to one. Seed is an unsigned 32-bit integer; counts are positive
    and K <= N. Tolerance is finite and nonnegative. Features must be scaled by
    the caller to define a meaningful Euclidean metric. Labels retain backend
    order; duplicate points can yield fewer distinct clusters. Neither global
    optimality nor cross-version numeric reproducibility is promised.
    """
    values = np.asarray(points)
    if (values.ndim != 2 or min(values.shape) == 0
            or values.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(values).all()):
        raise ValueError('Nonempty finite float32/float64 matrix required')
    count = _integer(n_clusters, 1, len(values))
    seed = _integer(random_state, 0, 2**32-1)
    restarts, iterations = _integer(n_init, 1), _integer(max_iter, 1)
    if (isinstance(tolerance, (bool, np.bool_))
            or not isinstance(tolerance, (int, float, np.floating))
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Finite nonnegative tolerance required')
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=count, init='k-means++', n_init=restarts,
                       max_iter=iterations, tol=float(tolerance), random_state=seed,
                       copy_x=True, algorithm='lloyd').fit(values.copy())
    if not np.isfinite(model.cluster_centers_).all():
        raise ValueError('Clustering produced nonfinite centers')
    return model.cluster_centers_.copy(), model.labels_.copy()
