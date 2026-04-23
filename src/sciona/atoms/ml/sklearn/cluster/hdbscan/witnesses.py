"""Ghost witnesses for limited HDBSCAN boundary atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


_VALID_METRICS = {"euclidean", "l2", "minkowski", "manhattan", "cityblock"}
_VALID_ALGORITHMS = {"auto", "brute", "kd_tree", "ball_tree"}
_VALID_SELECTION_METHODS = {"eom", "leaf"}


def _check_hdbscan_inputs(
    X: AbstractArray,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_epsilon: float,
    max_cluster_size: int | None,
    metric: str,
    alpha: float,
    algorithm: str,
    leaf_size: int,
    cluster_selection_method: str,
    allow_single_cluster: bool,
    n_jobs: int | None,
    copy: bool,
) -> tuple[int, int]:
    del allow_single_cluster, copy
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    if n_samples < 2 or n_features < 1:
        raise ValueError("HDBSCAN requires at least two samples and one feature")
    if min_cluster_size < 2 or min_cluster_size > n_samples:
        raise ValueError("min_cluster_size must be between two and sample count")
    if min_samples is not None and (min_samples < 1 or min_samples > n_samples):
        raise ValueError("min_samples must be None or between one and sample count")
    if cluster_selection_epsilon < 0.0:
        raise ValueError("cluster_selection_epsilon must be nonnegative")
    if max_cluster_size is not None and max_cluster_size < 0:
        raise ValueError("max_cluster_size must be nonnegative or None")
    if metric not in _VALID_METRICS:
        raise ValueError("unsupported HDBSCAN metric")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if algorithm not in _VALID_ALGORITHMS:
        raise ValueError("invalid HDBSCAN algorithm")
    if leaf_size < 1:
        raise ValueError("leaf_size must be positive")
    if cluster_selection_method not in _VALID_SELECTION_METHODS:
        raise ValueError("invalid cluster selection method")
    if n_jobs is not None and n_jobs == 0:
        raise ValueError("n_jobs must be nonzero or None")
    return n_samples, n_features


def witness_hdbscan_fit(
    X: AbstractArray,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    max_cluster_size: int | None = None,
    metric: str = "euclidean",
    metric_params: dict[str, float] | None = None,
    alpha: float = 1.0,
    algorithm: str = "auto",
    leaf_size: int = 40,
    n_jobs: int | None = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    copy: bool = True,
) -> AbstractArray:
    """Describe HDBSCAN labels from a fitted boundary state."""
    del metric_params
    n_samples, _ = _check_hdbscan_inputs(
        X,
        min_cluster_size,
        min_samples,
        cluster_selection_epsilon,
        max_cluster_size,
        metric,
        alpha,
        algorithm,
        leaf_size,
        cluster_selection_method,
        allow_single_cluster,
        n_jobs,
        copy,
    )
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)


def witness_hdbscan_fit_predict(
    X: AbstractArray,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    max_cluster_size: int | None = None,
    metric: str = "euclidean",
    metric_params: dict[str, float] | None = None,
    alpha: float = 1.0,
    algorithm: str = "auto",
    leaf_size: int = 40,
    n_jobs: int | None = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    copy: bool = True,
) -> AbstractArray:
    """Describe labels returned by HDBSCAN fit_predict."""
    return witness_hdbscan_fit(
        X,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
        metric=metric,
        metric_params=metric_params,
        alpha=alpha,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs=n_jobs,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        copy=copy,
    )
