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
