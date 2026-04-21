"""Ghost witnesses for sklearn manifold atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


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
