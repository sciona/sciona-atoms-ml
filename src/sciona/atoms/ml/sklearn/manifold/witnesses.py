"""Ghost witnesses for sklearn manifold atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import IsomapState, LocallyLinearEmbeddingState


def witness_classical_mds_dissimilarity_matrix(
    X: AbstractArray,
    *,
    metric: str = "euclidean",
    metric_params: None = None,
) -> AbstractArray:
    """Describe the dense pairwise dissimilarity matrix used by classical MDS."""
    del metric_params
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if metric == "precomputed":
        if X.shape[0] != X.shape[1]:
            raise ValueError("precomputed dissimilarities must be square")
        return AbstractArray(shape=(int(X.shape[0]), int(X.shape[0])), dtype="float64")
    if metric != "euclidean":
        raise ValueError("only euclidean or precomputed metrics are covered")
    return AbstractArray(shape=(int(X.shape[0]), int(X.shape[0])), dtype="float64")


def witness_classical_mds_double_center(dissimilarity_matrix: AbstractArray) -> AbstractArray:
    """Describe double-centering a square dissimilarity matrix."""
    if len(dissimilarity_matrix.shape) != 2:
        raise ValueError("dissimilarity_matrix must be 2D")
    if dissimilarity_matrix.shape[0] != dissimilarity_matrix.shape[1]:
        raise ValueError("dissimilarity_matrix must be square")
    return AbstractArray(shape=(int(dissimilarity_matrix.shape[0]), int(dissimilarity_matrix.shape[1])), dtype="float64")


def witness_classical_mds_fit(
    X: AbstractArray,
    *,
    n_components: int = 2,
    metric: str = "euclidean",
    metric_params: None = None,
) -> AbstractArray:
    """Describe fitting dense classical MDS coordinates."""
    del metric_params
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if n_components < 1 or n_components > X.shape[0]:
        raise ValueError("n_components must fit the sample count")
    if metric == "precomputed" and X.shape[0] != X.shape[1]:
        raise ValueError("precomputed dissimilarities must be square")
    if metric not in {"euclidean", "precomputed"}:
        raise ValueError("only euclidean or precomputed metrics are covered")
    return AbstractArray(shape=(int(X.shape[0]), n_components), dtype="float64")


def witness_smacof(
    dissimilarities: AbstractArray,
    *,
    metric: bool = True,
    n_components: int = 2,
    init: AbstractArray | None = None,
    n_init: int = 1,
    n_jobs: None = None,
    max_iter: int = 300,
    verbose: int | bool = 0,
    eps: float = 1e-6,
    random_state: int | None = None,
    return_n_iter: bool = False,
    normalized_stress: bool | str = "auto",
) -> AbstractArray:
    """Describe coordinates from metric stress majorization."""
    del n_jobs, verbose, eps, random_state, return_n_iter, normalized_stress
    if len(dissimilarities.shape) != 2 or dissimilarities.shape[0] != dissimilarities.shape[1]:
        raise ValueError("dissimilarities must be square")
    if metric is not True:
        raise ValueError("only metric SMACOF is covered")
    if n_components < 1 or n_components > dissimilarities.shape[0]:
        raise ValueError("n_components must fit the sample count")
    if init is not None and (len(init.shape) != 2 or init.shape[0] != dissimilarities.shape[0] or init.shape[1] != n_components):
        raise ValueError("init must match sample and component counts")
    if n_init != 1:
        raise ValueError("only n_init=1 is covered")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    return AbstractArray(shape=(int(dissimilarities.shape[0]), n_components), dtype="float64")


def witness_mds_fit(
    X: AbstractArray,
    *,
    n_components: int = 2,
    metric_mds: bool = True,
    n_init: int = 1,
    init: AbstractArray | None = None,
    max_iter: int = 300,
    verbose: int | bool = 0,
    eps: float = 1e-6,
    n_jobs: None = None,
    random_state: int | None = None,
    metric: str = "euclidean",
    metric_params: None = None,
    normalized_stress: bool | str = "auto",
) -> AbstractArray:
    """Describe fitting metric MDS coordinates with SMACOF."""
    del verbose, eps, n_jobs, random_state, metric_params, normalized_stress
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if metric_mds is not True:
        raise ValueError("only metric MDS is covered")
    if n_init != 1:
        raise ValueError("only n_init=1 is covered")
    if n_components < 1 or n_components > X.shape[0]:
        raise ValueError("n_components must fit the sample count")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if metric == "precomputed" and X.shape[0] != X.shape[1]:
        raise ValueError("precomputed dissimilarities must be square")
    if metric not in {"euclidean", "precomputed"}:
        raise ValueError("only euclidean or precomputed metrics are covered")
    if init is not None and (len(init.shape) != 2 or init.shape[0] != X.shape[0] or init.shape[1] != n_components):
        raise ValueError("init must match sample and component counts")
    return AbstractArray(shape=(int(X.shape[0]), n_components), dtype="float64")


def witness_spectral_embedding(
    adjacency: AbstractArray,
    *,
    n_components: int = 8,
    eigen_solver: str | None = None,
    random_state: int | None = None,
    eigen_tol: float | str = "auto",
    norm_laplacian: bool = True,
    drop_first: bool = True,
) -> AbstractArray:
    """Describe dense graph Laplacian eigenmap coordinates."""
    del random_state
    if len(adjacency.shape) != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if (
        not isinstance(n_components, int)
        or isinstance(n_components, bool)
        or n_components < 1
        or n_components >= adjacency.shape[0]
    ):
        raise ValueError("n_components must be positive and below sample count")
    if eigen_solver not in {None, "arpack"}:
        raise ValueError("only arpack/default eigen solving is covered")
    if eigen_tol != "auto" and (
        not isinstance(eigen_tol, (int, float)) or isinstance(eigen_tol, bool) or eigen_tol < 0
    ):
        raise ValueError("eigen_tol must be non-negative or auto")
    if not isinstance(norm_laplacian, bool):
        raise ValueError("norm_laplacian must be boolean")
    if not isinstance(drop_first, bool):
        raise ValueError("drop_first must be boolean")
    return AbstractArray(shape=(int(adjacency.shape[0]), n_components), dtype="float64")


def witness_spectral_embedding_fit(
    X: AbstractArray,
    *,
    n_components: int = 2,
    affinity: str = "rbf",
    gamma: float | None = None,
    random_state: int | None = None,
    eigen_solver: str | None = None,
    eigen_tol: float | str = "auto",
    n_neighbors: None = None,
    n_jobs: None = None,
) -> AbstractArray:
    """Describe fitting dense spectral embedding coordinates."""
    del random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if (
        not isinstance(n_components, int)
        or isinstance(n_components, bool)
        or n_components < 1
        or n_components >= X.shape[0]
    ):
        raise ValueError("n_components must be positive and below sample count")
    if affinity not in {"rbf", "precomputed"}:
        raise ValueError("only rbf and precomputed affinities are covered")
    if affinity == "precomputed" and X.shape[0] != X.shape[1]:
        raise ValueError("precomputed affinity must be square")
    if gamma is not None and (not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or gamma <= 0):
        raise ValueError("gamma must be positive when provided")
    if eigen_solver not in {None, "arpack"}:
        raise ValueError("only arpack/default eigen solving is covered")
    if eigen_tol != "auto" and (
        not isinstance(eigen_tol, (int, float)) or isinstance(eigen_tol, bool) or eigen_tol < 0
    ):
        raise ValueError("eigen_tol must be non-negative or auto")
    if n_neighbors is not None:
        raise ValueError("nearest-neighbor affinity is outside this atom scope")
    if n_jobs is not None:
        raise ValueError("parallel neighbor construction is outside this atom scope")
    return AbstractArray(shape=(int(X.shape[0]), n_components), dtype="float64")


def witness_isomap_neighbors_graph(
    X: AbstractArray,
    *,
    n_neighbors: int = 5,
    radius: None = None,
    neighbors_algorithm: str = "auto",
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
    n_jobs: None = None,
) -> AbstractArray:
    """Describe the dense distance graph used by the Isomap fit path."""
    del radius, neighbors_algorithm, metric, p, metric_params, n_jobs
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if n_neighbors >= X.shape[0]:
        raise ValueError("n_neighbors must be below sample count")
    return AbstractArray(shape=(int(X.shape[0]), int(X.shape[0])), dtype="float64")


def witness_isomap_geodesic_distances(
    neighbors_graph: AbstractArray,
    *,
    path_method: str = "auto",
) -> AbstractArray:
    """Describe all-pairs shortest paths over an Isomap neighbor graph."""
    del path_method
    if len(neighbors_graph.shape) != 2 or neighbors_graph.shape[0] != neighbors_graph.shape[1]:
        raise ValueError("neighbors_graph must be square")
    return AbstractArray(shape=(int(neighbors_graph.shape[0]), int(neighbors_graph.shape[1])), dtype="float64")


def witness_isomap_fit(
    X: AbstractArray,
    *,
    n_neighbors: int = 5,
    radius: None = None,
    n_components: int = 2,
    eigen_solver: str = "dense",
    tol: float = 0.0,
    max_iter: None = None,
    path_method: str = "auto",
    neighbors_algorithm: str = "auto",
    n_jobs: None = None,
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
) -> AbstractArray:
    """Describe fitting dense Isomap coordinates."""
    del radius, eigen_solver, tol, max_iter, path_method, neighbors_algorithm, n_jobs, metric, p, metric_params
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be positive")
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if n_neighbors >= X.shape[0]:
        raise ValueError("n_neighbors must be below sample count")
    return AbstractArray(shape=(int(X.shape[0]), n_components), dtype="float64")


def witness_isomap_transform(X: AbstractArray, state: IsomapState) -> AbstractArray:
    """Describe transforming query samples through a fitted Isomap state."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted Isomap state")
    return AbstractArray(shape=(int(X.shape[0]), state.n_components), dtype="float64")


def witness_isomap_reconstruction_error(state: IsomapState) -> AbstractArray:
    """Describe the scalar Isomap reconstruction error."""
    del state
    return AbstractArray(shape=(), dtype="float64")


def witness_lle_barycenter_weights(
    X: AbstractArray,
    Y: AbstractArray,
    indices: AbstractArray,
    *,
    reg: float = 1e-3,
) -> AbstractArray:
    """Describe local reconstruction weights for LLE neighbors."""
    if len(X.shape) != 2 or len(Y.shape) != 2 or len(indices.shape) != 2:
        raise ValueError("X, Y, and indices must be 2D")
    if X.shape[0] != indices.shape[0]:
        raise ValueError("indices must have one row per X sample")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same feature count")
    if indices.shape[1] < 1:
        raise ValueError("indices must select at least one neighbor")
    if not isinstance(reg, (int, float)) or isinstance(reg, bool) or reg < 0:
        raise ValueError("reg must be nonnegative")
    return AbstractArray(shape=(int(X.shape[0]), int(indices.shape[1])), dtype="float64")


def witness_lle_barycenter_graph(
    X: AbstractArray,
    *,
    n_neighbors: int,
    reg: float = 1e-3,
    n_jobs: None = None,
) -> AbstractArray:
    """Describe the dense LLE barycenter weight matrix."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if n_neighbors >= X.shape[0]:
        raise ValueError("n_neighbors must be below sample count")
    if not isinstance(reg, (int, float)) or isinstance(reg, bool) or reg < 0:
        raise ValueError("reg must be nonnegative")
    if n_jobs is not None:
        raise ValueError("parallel neighbor search is outside this atom scope")
    return AbstractArray(shape=(int(X.shape[0]), int(X.shape[0])), dtype="float64")


def witness_lle_standard_reconstruction_matrix(weights: AbstractArray) -> AbstractArray:
    """Describe the standard LLE reconstruction matrix."""
    if len(weights.shape) != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be square")
    return AbstractArray(shape=(int(weights.shape[0]), int(weights.shape[1])), dtype="float64")


def witness_locally_linear_embedding(
    X: AbstractArray,
    *,
    n_neighbors: int,
    n_components: int,
    reg: float = 1e-3,
    eigen_solver: str = "dense",
    tol: float = 1e-6,
    max_iter: int = 100,
    method: str = "standard",
    hessian_tol: float = 1e-4,
    modified_tol: float = 1e-12,
    random_state: int | None = None,
    n_jobs: None = None,
) -> AbstractArray:
    """Describe fitting standard dense locally linear embedding coordinates."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if n_neighbors >= X.shape[0]:
        raise ValueError("n_neighbors must be below sample count")
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be positive")
    if n_components > X.shape[1]:
        raise ValueError("n_components must not exceed feature count")
    if not isinstance(reg, (int, float)) or isinstance(reg, bool) or reg < 0:
        raise ValueError("reg must be nonnegative")
    if eigen_solver != "dense":
        raise ValueError("only dense eigen solving is covered")
    if not isinstance(tol, (int, float)) or isinstance(tol, bool) or tol < 0:
        raise ValueError("tol must be nonnegative")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be positive")
    if method != "standard":
        raise ValueError("only standard LLE is covered")
    if not isinstance(hessian_tol, (int, float)) or isinstance(hessian_tol, bool) or hessian_tol < 0:
        raise ValueError("hessian_tol must be nonnegative")
    if not isinstance(modified_tol, (int, float)) or isinstance(modified_tol, bool) or modified_tol < 0:
        raise ValueError("modified_tol must be nonnegative")
    if random_state is not None and (not isinstance(random_state, int) or isinstance(random_state, bool)):
        raise ValueError("random_state must be an integer or None")
    if n_jobs is not None:
        raise ValueError("parallel neighbor search is outside this atom scope")
    return AbstractArray(shape=(int(X.shape[0]), n_components), dtype="float64")


def witness_locally_linear_embedding_fit(
    X: AbstractArray,
    *,
    n_neighbors: int = 5,
    n_components: int = 2,
    reg: float = 1e-3,
    eigen_solver: str = "dense",
    tol: float = 1e-6,
    max_iter: int = 100,
    method: str = "standard",
    hessian_tol: float = 1e-4,
    modified_tol: float = 1e-12,
    neighbors_algorithm: str = "auto",
    random_state: int | None = None,
    n_jobs: None = None,
) -> AbstractArray:
    """Describe fitting a standard dense LLE estimator state."""
    if neighbors_algorithm not in {"auto", "brute", "kd_tree", "ball_tree"}:
        raise ValueError("unsupported neighbors_algorithm")
    return witness_locally_linear_embedding(
        X,
        n_neighbors=n_neighbors,
        n_components=n_components,
        reg=reg,
        eigen_solver=eigen_solver,
        tol=tol,
        max_iter=max_iter,
        method=method,
        hessian_tol=hessian_tol,
        modified_tol=modified_tol,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def witness_locally_linear_embedding_transform(
    X: AbstractArray,
    state: LocallyLinearEmbeddingState,
) -> AbstractArray:
    """Describe transforming samples with a fitted standard dense LLE state."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted LLE state")
    return AbstractArray(shape=(int(X.shape[0]), state.n_components), dtype="float64")
