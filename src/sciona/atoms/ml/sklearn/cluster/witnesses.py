"""Ghost witnesses for selected sklearn cluster atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import AffinityPropagationState, MeanShiftState


def witness_affinity_propagation(
    S: AbstractArray,
    *,
    preference: float | AbstractArray | None = None,
    convergence_iter: int = 15,
    max_iter: int = 200,
    damping: float = 0.5,
    copy: bool = True,
    verbose: bool = False,
    return_n_iter: bool = False,
    random_state: int | None = None,
) -> tuple[AbstractArray, AbstractArray] | tuple[AbstractArray, AbstractArray, int]:
    """Describe cluster centers and labels from a similarity matrix."""
    del preference, copy, verbose, random_state
    n_samples = _check_square(S)
    _check_iteration_parameters(convergence_iter, max_iter, damping)
    centers = AbstractArray(shape=(n_samples,), dtype="int64", min_val=0)
    labels = AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)
    if return_n_iter:
        return centers, labels, max_iter
    return centers, labels


def witness_affinity_propagation_fit(
    X: AbstractArray,
    *,
    damping: float = 0.5,
    max_iter: int = 200,
    convergence_iter: int = 15,
    copy: bool = True,
    preference: float | AbstractArray | None = None,
    affinity: str = "euclidean",
    verbose: bool = False,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe fitting affinity propagation into immutable state."""
    del copy, preference, verbose, random_state
    n_samples, _ = _check_2d(X)
    if affinity not in {"euclidean", "precomputed"}:
        raise ValueError("affinity must be 'euclidean' or 'precomputed'")
    if affinity == "precomputed" and X.shape[0] != X.shape[1]:
        raise ValueError("precomputed affinity must be square")
    _check_iteration_parameters(convergence_iter, max_iter, damping)
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)


def witness_affinity_propagation_predict(
    X: AbstractArray,
    state: AffinityPropagationState,
) -> AbstractArray:
    """Describe nearest-center prediction from fitted affinity propagation state."""
    n_samples, n_features = _check_2d(X)
    if state.affinity == "precomputed":
        raise ValueError("predict is not supported for precomputed affinity")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)


def witness_estimate_bandwidth(
    X: AbstractArray,
    *,
    quantile: float = 0.3,
    n_samples: int | None = None,
    random_state: int | None = 0,
    n_jobs: int | None = None,
) -> float:
    """Describe estimating a nonnegative mean-shift bandwidth."""
    del random_state, n_jobs
    n_rows, _ = _check_2d(X)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if n_samples is not None and n_samples < 1:
        raise ValueError("n_samples must be positive or None")
    if n_rows < 1:
        raise ValueError("X must contain samples")
    return 0.0


def witness_mean_shift(
    X: AbstractArray,
    *,
    bandwidth: float | None = None,
    seeds: AbstractArray | None = None,
    bin_seeding: bool = False,
    min_bin_freq: int = 1,
    cluster_all: bool = True,
    max_iter: int = 300,
    n_jobs: int | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe mean-shift centers and labels for input samples."""
    del bin_seeding, cluster_all, n_jobs
    n_samples, n_features = _check_mean_shift_inputs(X, bandwidth, seeds, min_bin_freq, max_iter)
    centers = AbstractArray(shape=(n_samples, n_features), dtype="float64")
    labels = AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)
    return centers, labels


def witness_mean_shift_fit(
    X: AbstractArray,
    *,
    bandwidth: float | None = None,
    seeds: AbstractArray | None = None,
    bin_seeding: bool = False,
    min_bin_freq: int = 1,
    cluster_all: bool = True,
    max_iter: int = 300,
    n_jobs: int | None = None,
) -> AbstractArray:
    """Describe fitting mean-shift cluster centers and labels."""
    del bin_seeding, cluster_all, n_jobs
    n_samples, n_features = _check_mean_shift_inputs(X, bandwidth, seeds, min_bin_freq, max_iter)
    return AbstractArray(shape=(n_samples, n_features), dtype="float64")


def witness_mean_shift_predict(
    X: AbstractArray,
    state: MeanShiftState,
) -> AbstractArray:
    """Describe nearest-center predictions from fitted mean-shift state."""
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=0)


def witness_kmeans_plusplus(
    X: AbstractArray,
    n_clusters: int,
    *,
    sample_weight: AbstractArray | None = None,
    x_squared_norms: AbstractArray | None = None,
    random_state: int | None = None,
    n_local_trials: int | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe k-means++ seed centers and source row indices."""
    del random_state
    n_samples, n_features = _check_2d(X)
    if n_clusters < 1 or n_clusters > n_samples:
        raise ValueError("n_clusters must be between one and sample count")
    if sample_weight is not None and sample_weight.shape != (n_samples,):
        raise ValueError("sample_weight must match sample count")
    if x_squared_norms is not None and x_squared_norms.shape != (n_samples,):
        raise ValueError("x_squared_norms must match sample count")
    if n_local_trials is not None and n_local_trials < 1:
        raise ValueError("n_local_trials must be positive or None")
    centers = AbstractArray(shape=(n_clusters, n_features), dtype="float64")
    indices = AbstractArray(shape=(n_clusters,), dtype="int64", min_val=0)
    return centers, indices


def witness_cluster_optics_dbscan(
    *,
    reachability: AbstractArray,
    core_distances: AbstractArray,
    ordering: AbstractArray,
    eps: float,
) -> AbstractArray:
    """Describe DBSCAN-style labels extracted from an OPTICS ordering."""
    n_samples = _check_matching_optics_vectors(reachability, core_distances, ordering)
    if eps < 0.0:
        raise ValueError("eps must be nonnegative")
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)


def witness_cluster_optics_xi(
    *,
    reachability: AbstractArray,
    predecessor: AbstractArray,
    ordering: AbstractArray,
    min_samples: int | float,
    min_cluster_size: int | float | None = None,
    xi: float = 0.05,
    predecessor_correction: bool = True,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe Xi-steep labels and cluster intervals from OPTICS arrays."""
    del predecessor_correction
    n_samples = _check_matching_optics_vectors(reachability, predecessor, ordering)
    _check_fraction_or_count(min_samples, n_samples, "min_samples")
    if min_cluster_size is not None:
        _check_fraction_or_count(min_cluster_size, n_samples, "min_cluster_size")
    if not 0.0 <= xi <= 1.0:
        raise ValueError("xi must be in [0, 1]")
    labels = AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)
    clusters = AbstractArray(shape=(n_samples, 2), dtype="int64", min_val=0)
    return labels, clusters


def witness_compute_optics_graph(
    X: AbstractArray,
    *,
    min_samples: int | float,
    max_eps: float,
    metric: str,
    p: float | None,
    metric_params: dict[str, float] | None,
    algorithm: str,
    leaf_size: int,
    n_jobs: int | None,
) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    """Describe ordered reachability arrays for density clustering."""
    del metric, p, metric_params, n_jobs
    n_samples, _ = _check_2d(X)
    _check_fraction_or_count(min_samples, n_samples, "min_samples")
    if max_eps < 0.0:
        raise ValueError("max_eps must be nonnegative")
    if algorithm not in {"auto", "brute", "ball_tree", "kd_tree"}:
        raise ValueError("invalid neighbor algorithm")
    if leaf_size < 1:
        raise ValueError("leaf_size must be positive")
    ordering = AbstractArray(shape=(n_samples,), dtype="int64", min_val=0)
    core_distances = AbstractArray(shape=(n_samples,), dtype="float64", min_val=0.0)
    reachability = AbstractArray(shape=(n_samples,), dtype="float64", min_val=0.0)
    predecessor = AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)
    return ordering, core_distances, reachability, predecessor


def _check_2d(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return int(X.shape[0]), int(X.shape[1])


def _check_square(S: AbstractArray) -> int:
    rows, cols = _check_2d(S)
    if rows != cols:
        raise ValueError("similarity matrix must be square")
    return rows


def _check_iteration_parameters(convergence_iter: int, max_iter: int, damping: float) -> None:
    if convergence_iter < 1:
        raise ValueError("convergence_iter must be at least one")
    if max_iter < 1:
        raise ValueError("max_iter must be at least one")
    if not 0.5 <= damping < 1.0:
        raise ValueError("damping must be in [0.5, 1.0)")


def _check_mean_shift_inputs(
    X: AbstractArray,
    bandwidth: float | None,
    seeds: AbstractArray | None,
    min_bin_freq: int,
    max_iter: int,
) -> tuple[int, int]:
    n_samples, n_features = _check_2d(X)
    if bandwidth is not None and bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive or None")
    if seeds is not None:
        seed_samples, seed_features = _check_2d(seeds)
        if seed_samples < 1 or seed_features != n_features:
            raise ValueError("seeds must be 2D with matching feature count")
    if min_bin_freq < 1:
        raise ValueError("min_bin_freq must be at least one")
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative")
    return n_samples, n_features


def _check_matching_optics_vectors(
    reachability: AbstractArray,
    core_distances: AbstractArray,
    ordering: AbstractArray,
) -> int:
    if len(reachability.shape) != 1:
        raise ValueError("reachability must be 1D")
    if len(core_distances.shape) != 1:
        raise ValueError("core_distances must be 1D")
    if len(ordering.shape) != 1:
        raise ValueError("ordering must be 1D")
    n_samples = int(reachability.shape[0])
    if int(core_distances.shape[0]) != n_samples or int(ordering.shape[0]) != n_samples:
        raise ValueError("OPTICS vectors must have equal length")
    return n_samples


def _check_fraction_or_count(value: int | float, n_samples: int, name: str) -> None:
    if isinstance(value, int):
        if value < 2 or value > n_samples:
            raise ValueError(f"{name} must be between 2 and sample count")
        return
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} fraction must be in [0, 1]")
