"""Reusable fitted Euclidean search preserving scikit-learn backend ordering.

Unlike the dense search atoms, these operations retain the fitted auto-selected
backend. Row indices refer only to caller-supplied points; no domain identifiers,
partitioning, projections or downstream ranking policy are embedded.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete fitted search and finite numeric points required')


def _points(points):
    value = np.asarray(points)
    if value.dtype != np.float64 or value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError('Nonempty finite float64 matrix required')
    return value


def _positive_int(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError('Positive integer required')
    return int(value)


def _state(state):
    if not isinstance(state, NearestNeighbors) or not hasattr(state, 'n_samples_fit_'):
        raise ValueError('Fitted Euclidean search state required')
    if (state.algorithm != 'auto' or state.metric != 'minkowski' or state.p != 2
            or state.metric_params is not None or state.n_jobs != 1):
        raise ValueError('Unmodified serial Euclidean search configuration required')
    return state


def _finite_result(result):
    distances, indices = result
    if any(not np.isfinite(row).all() for row in distances):
        raise ValueError('Search produced nonfinite distances')
    return distances, indices


@register_atom(concrete_only)
def fit_euclidean_search(points: np.ndarray, default_neighbors: int = 5,
                         leaf_size: int = 30) -> object:
    """Fit a reusable index from nonempty finite float64 points (N,D).

    Copies caller storage. Default neighbor count affects auto backend selection
    even when a later query overrides k; it may exceed N at fit time. Queries
    require k <= N. State is an opaque runtime object, must not be modified, and
    is not a portable serialized artifact. Uses one worker and one native thread.
    """
    values = _points(points).copy()
    model = NearestNeighbors(n_neighbors=_positive_int(default_neighbors),
        leaf_size=_positive_int(leaf_size), n_jobs=1)
    with threadpool_limits(limits=1):
        return model.fit(values)


@register_atom(concrete_only)
def query_euclidean_k(points: np.ndarray, state: object, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return distances and fitted-row indices, each (Q,k), for explicit queries.

    Queries use the same feature units as fitted points. Explicit self queries
    include self. Backend tie selection and ordering are preserved, with no
    cross-version ordering guarantee. Empty queries reject; caller may skip them.
    """
    values, model, count = _points(points), _state(state), _positive_int(k)
    if count > model.n_samples_fit_:
        raise ValueError('Neighbor count exceeds fitted points')
    with threadpool_limits(limits=1):
        return _finite_result(model.kneighbors(values, n_neighbors=count))


@register_atom(concrete_only)
def query_euclidean_radius(points: np.ndarray, state: object,
                           radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Return unsorted ragged distance/index rows within an inclusive radius.

    Radius shares the feature unit. Zero radius and rows with no matches are
    supported. Unsorted backend order is retained; no input-order sorting is
    added. Finite inputs yielding nonfinite distances reject.
    """
    values, model = _points(points), _state(state)
    if (isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, float, np.floating))
            or not np.isfinite(radius) or radius < 0):
        raise ValueError('Finite nonnegative radius required')
    with threadpool_limits(limits=1):
        return _finite_result(model.radius_neighbors(values, radius=radius, sort_results=False))
