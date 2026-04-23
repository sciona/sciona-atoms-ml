"""Ghost witnesses for limited DBSCAN boundary atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_2d(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return int(X.shape[0]), int(X.shape[1])


def _check_dbscan_inputs(
    X: AbstractArray,
    eps: float,
    min_samples: int,
    metric: str,
    algorithm: str,
    leaf_size: int,
    p: float | None,
    sample_weight: AbstractArray | None,
    n_jobs: int | None,
) -> tuple[int, int]:
    n_samples, n_features = _check_2d(X)
    if metric == "precomputed" and n_samples != n_features:
        raise ValueError("precomputed DBSCAN input must be square")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    if metric not in {"euclidean", "minkowski", "manhattan", "precomputed"}:
        raise ValueError("unsupported DBSCAN metric")
    if algorithm not in {"auto", "ball_tree", "kd_tree", "brute"}:
        raise ValueError("invalid neighbor algorithm")
    if leaf_size < 1:
        raise ValueError("leaf_size must be positive")
    if p is not None and p <= 0.0:
        raise ValueError("p must be positive or None")
    if sample_weight is not None and sample_weight.shape != (n_samples,):
        raise ValueError("sample_weight must match sample count")
    if n_jobs is not None and n_jobs == 0:
        raise ValueError("n_jobs must be nonzero or None")
    return n_samples, n_features


def witness_dbscan_fit(
    X: AbstractArray,
    *,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "minkowski",
    metric_params: dict[str, float] | None = None,
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: float | None = 2.0,
    sample_weight: AbstractArray | None = None,
    n_jobs: int | None = None,
) -> AbstractArray:
    """Describe DBSCAN labels from a fitted boundary state."""
    del metric_params
    n_samples, _ = _check_dbscan_inputs(X, eps, min_samples, metric, algorithm, leaf_size, p, sample_weight, n_jobs)
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)


def witness_dbscan_core_labels(
    X: AbstractArray,
    *,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "minkowski",
    metric_params: dict[str, float] | None = None,
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: float | None = 2.0,
    sample_weight: AbstractArray | None = None,
    n_jobs: int | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe public DBSCAN helper core indices and labels."""
    del metric_params
    n_samples, _ = _check_dbscan_inputs(X, eps, min_samples, metric, algorithm, leaf_size, p, sample_weight, n_jobs)
    core_indices = AbstractArray(shape=(n_samples,), dtype="int64", min_val=0)
    labels = AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)
    return core_indices, labels
