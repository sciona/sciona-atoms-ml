"""Manifold-learning atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.sparse.csgraph import connected_components, laplacian as csgraph_laplacian
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh

from sciona.ghost.registry import register_atom

from .state_models import (
    ClassicalMDSState,
    IsomapState,
    LocallyLinearEmbeddingState,
    MDSState,
    SMACOFState,
    SpectralEmbeddingState,
)
from .witnesses import (
    witness_classical_mds_dissimilarity_matrix,
    witness_classical_mds_double_center,
    witness_classical_mds_fit,
    witness_isomap_fit,
    witness_isomap_geodesic_distances,
    witness_isomap_neighbors_graph,
    witness_isomap_reconstruction_error,
    witness_isomap_transform,
    witness_lle_barycenter_graph,
    witness_lle_barycenter_weights,
    witness_lle_standard_reconstruction_matrix,
    witness_locally_linear_embedding,
    witness_locally_linear_embedding_fit,
    witness_locally_linear_embedding_transform,
    witness_mds_fit,
    witness_smacof,
    witness_spectral_embedding,
    witness_spectral_embedding_fit,
)

def _matrix_2d(X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2)

def _square_matrix(matrix: NDArray[np.float64]) -> bool:
    values = np.asarray(matrix)
    return bool(values.ndim == 2 and values.shape[0] == values.shape[1])

def _finite_matrix(matrix: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))

def _metric_valid(metric: str) -> bool:
    return metric in {"euclidean", "precomputed"}

def _metric_params_valid(metric_params: None) -> bool:
    return metric_params is None

def _precomputed_shape_valid(X: NDArray[np.float64], metric: str) -> bool:
    return bool(metric != "precomputed" or _square_matrix(X))

def _n_components_valid(n_components: int, X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(isinstance(n_components, int) and not isinstance(n_components, bool) and values.ndim == 2 and 1 <= n_components <= values.shape[0])

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _positive_finite(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)

def _random_state_valid(random_state: int | None) -> bool:
    return bool(random_state is None or (isinstance(random_state, int) and not isinstance(random_state, bool)))

def _init_shape_valid(init: NDArray[np.float64] | None, n_samples: int, n_components: int) -> bool:
    if init is None:
        return True
    values = np.asarray(init)
    return bool(values.ndim == 2 and values.shape == (n_samples, n_components) and np.all(np.isfinite(values)))

def _normalized_stress_valid(normalized_stress: bool | str) -> bool:
    return bool(isinstance(normalized_stress, bool) or normalized_stress == "auto")

def _eigen_solver_valid(eigen_solver: str | None) -> bool:
    return bool(eigen_solver in {None, "arpack"})

def _eigen_tol_valid(eigen_tol: float | str) -> bool:
    return bool(eigen_tol == "auto" or (isinstance(eigen_tol, (int, float)) and not isinstance(eigen_tol, bool) and np.isfinite(float(eigen_tol)) and float(eigen_tol) >= 0.0))

def _spectral_components_valid(n_components: int, adjacency: NDArray[np.float64]) -> bool:
    values = np.asarray(adjacency)
    return bool(isinstance(n_components, int) and not isinstance(n_components, bool) and values.ndim == 2 and 1 <= n_components < values.shape[0])

def _spectral_affinity_valid(affinity: str) -> bool:
    return affinity in {"rbf", "precomputed"}

def _spectral_gamma_valid(gamma: float | None) -> bool:
    return bool(gamma is None or _positive_finite(gamma))

def _spectral_optional_none(value: None) -> bool:
    return value is None

def _dissimilarity_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_samples = np.asarray(X).shape[0]
    return bool(values.shape == (n_samples, n_samples) and np.all(np.isfinite(values)) and np.allclose(values, values.T))

def _centered_matrix_valid(result: NDArray[np.float64], dissimilarity_matrix: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(dissimilarity_matrix)
    return bool(
        values.shape == source.shape
        and np.all(np.isfinite(values))
        and np.allclose(values, values.T)
        and np.allclose(np.mean(values, axis=0), 0.0)
        and np.allclose(np.mean(values, axis=1), 0.0)
    )

def _classical_mds_state_valid(state: ClassicalMDSState) -> bool:
    n_samples = state.dissimilarity_matrix.shape[0]
    return bool(
        state.embedding.shape == (n_samples, state.n_components)
        and state.dissimilarity_matrix.shape == (n_samples, n_samples)
        and state.eigenvalues.shape == (state.n_components,)
        and state.n_components >= 1
        and state.n_features_in >= 1
        and _metric_valid(state.metric)
        and np.all(np.isfinite(state.embedding))
        and np.all(np.isfinite(state.dissimilarity_matrix))
        and np.all(np.isfinite(state.eigenvalues))
        and np.allclose(state.dissimilarity_matrix, state.dissimilarity_matrix.T)
        and np.all(state.eigenvalues >= -1e-10)
    )

def _smacof_state_valid(state: SMACOFState) -> bool:
    n_samples = state.dissimilarity_matrix.shape[0]
    return bool(
        state.embedding.shape == (n_samples, state.n_components)
        and state.dissimilarity_matrix.shape == (n_samples, n_samples)
        and state.n_components >= 1
        and state.n_iter >= 1
        and np.isfinite(state.stress)
        and state.stress >= 0.0
        and np.all(np.isfinite(state.embedding))
        and np.all(np.isfinite(state.dissimilarity_matrix))
        and np.allclose(state.dissimilarity_matrix, state.dissimilarity_matrix.T)
    )

def _mds_state_valid(state: MDSState) -> bool:
    n_samples = state.dissimilarity_matrix.shape[0]
    return bool(
        state.embedding.shape == (n_samples, state.n_components)
        and state.dissimilarity_matrix.shape == (n_samples, n_samples)
        and state.n_components >= 1
        and state.n_features_in >= 1
        and state.metric in {"euclidean", "precomputed"}
        and state.metric_mds is True
        and state.n_iter >= 1
        and np.isfinite(state.stress)
        and state.stress >= 0.0
        and np.all(np.isfinite(state.embedding))
        and np.all(np.isfinite(state.dissimilarity_matrix))
        and np.allclose(state.dissimilarity_matrix, state.dissimilarity_matrix.T)
    )

def _spectral_embedding_valid(result: NDArray[np.float64], adjacency: NDArray[np.float64], n_components: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(adjacency)
    return bool(values.shape == (source.shape[0], n_components) and np.all(np.isfinite(values)))

def _spectral_state_valid(state: SpectralEmbeddingState) -> bool:
    n_samples = state.affinity_matrix.shape[0]
    return bool(
        state.embedding.shape == (n_samples, state.n_components)
        and state.affinity_matrix.shape == (n_samples, n_samples)
        and state.n_components >= 1
        and state.affinity in {"rbf", "precomputed"}
        and state.eigen_solver == "arpack"
        and state.n_features_in >= 1
        and np.all(np.isfinite(state.embedding))
        and np.all(np.isfinite(state.affinity_matrix))
        and np.allclose(state.affinity_matrix, state.affinity_matrix.T)
    )

def _isomap_neighbors_valid(n_neighbors: int, X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(
        isinstance(n_neighbors, int)
        and not isinstance(n_neighbors, bool)
        and values.ndim == 2
        and 1 <= n_neighbors < values.shape[0]
    )

def _isomap_components_valid(n_components: int, X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(
        isinstance(n_components, int)
        and not isinstance(n_components, bool)
        and values.ndim == 2
        and 1 <= n_components <= values.shape[0]
    )

def _isomap_options_valid(
    radius: None,
    eigen_solver: str,
    tol: float,
    max_iter: None,
    path_method: str,
    neighbors_algorithm: str,
    n_jobs: None,
    metric: str,
    p: float,
    metric_params: None,
) -> bool:
    return bool(
        radius is None
        and eigen_solver == "dense"
        and isinstance(tol, (int, float))
        and not isinstance(tol, bool)
        and float(tol) >= 0.0
        and max_iter is None
        and path_method in {"auto", "FW", "D"}
        and neighbors_algorithm in {"auto", "brute", "kd_tree", "ball_tree"}
        and n_jobs is None
        and metric == "minkowski"
        and isinstance(p, (int, float))
        and not isinstance(p, bool)
        and float(p) >= 1.0
        and metric_params is None
    )

def _isomap_neighbor_options_valid(
    radius: None,
    neighbors_algorithm: str,
    n_jobs: None,
    metric: str,
    p: float,
    metric_params: None,
) -> bool:
    return bool(
        radius is None
        and neighbors_algorithm in {"auto", "brute", "kd_tree", "ball_tree"}
        and n_jobs is None
        and metric == "minkowski"
        and isinstance(p, (int, float))
        and not isinstance(p, bool)
        and float(p) >= 1.0
        and metric_params is None
    )

def _isomap_graph_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    graph = np.asarray(result, dtype=np.float64)
    n_samples = np.asarray(X).shape[0]
    return bool(
        graph.shape == (n_samples, n_samples)
        and np.all(np.isfinite(graph))
        and np.all(graph >= 0.0)
    )

def _isomap_distances_valid(result: NDArray[np.float64], neighbors_graph: NDArray[np.float64]) -> bool:
    distances = np.asarray(result, dtype=np.float64)
    graph = np.asarray(neighbors_graph)
    return bool(
        distances.shape == graph.shape
        and distances.shape[0] == distances.shape[1]
        and np.all(np.isfinite(distances))
        and np.allclose(distances, distances.T)
        and np.all(distances >= 0.0)
    )

def _isomap_state_valid(state: IsomapState) -> bool:
    n_samples = state.training_data.shape[0]
    return bool(
        state.embedding.shape == (n_samples, state.n_components)
        and state.dist_matrix.shape == (n_samples, n_samples)
        and state.eigenvalues.shape == (state.n_components,)
        and state.eigenvectors.shape == (n_samples, state.n_components)
        and state.kernel_centerer_rows.shape == (n_samples,)
        and state.training_data.shape[1] == state.n_features_in
        and 1 <= state.n_neighbors < n_samples
        and state.path_method in {"auto", "FW", "D"}
        and state.metric == "minkowski"
        and state.p >= 1.0
        and np.all(np.isfinite(state.embedding))
        and np.all(np.isfinite(state.dist_matrix))
        and np.all(np.isfinite(state.training_data))
        and np.all(np.isfinite(state.eigenvalues))
        and np.all(np.isfinite(state.eigenvectors))
        and np.all(np.isfinite(state.kernel_centerer_rows))
        and np.isfinite(state.kernel_centerer_all)
        and np.all(state.eigenvalues >= 0.0)
        and np.allclose(state.dist_matrix, state.dist_matrix.T)
    )

def _isomap_feature_count_matches(X: NDArray[np.float64], state: IsomapState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in)

def _isomap_transform_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: IsomapState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (np.asarray(X).shape[0], state.n_components) and np.all(np.isfinite(values)))

def _nonnegative_finite_scalar(value: float) -> bool:
    return bool(np.isfinite(float(value)) and float(value) >= 0.0)

def _integer_index_matrix(indices: NDArray[np.int64]) -> bool:
    values = np.asarray(indices)
    return bool(values.ndim == 2 and np.issubdtype(values.dtype, np.integer))

def _lle_indices_valid(X: NDArray[np.float64], Y: NDArray[np.float64], indices: NDArray[np.int64]) -> bool:
    x_values = np.asarray(X)
    y_values = np.asarray(Y)
    idx = np.asarray(indices)
    return bool(
        x_values.ndim == 2
        and y_values.ndim == 2
        and idx.ndim == 2
        and x_values.shape[0] == idx.shape[0]
        and x_values.shape[1] == y_values.shape[1]
        and idx.shape[1] >= 1
        and np.issubdtype(idx.dtype, np.integer)
        and np.all(idx >= 0)
        and np.all(idx < y_values.shape[0])
    )

def _lle_reg_valid(reg: float) -> bool:
    return bool(isinstance(reg, (int, float)) and not isinstance(reg, bool) and np.isfinite(float(reg)) and float(reg) >= 0.0)

def _lle_neighbors_valid(n_neighbors: int, X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(
        isinstance(n_neighbors, int)
        and not isinstance(n_neighbors, bool)
        and values.ndim == 2
        and 1 <= n_neighbors < values.shape[0]
    )

def _lle_components_valid(n_components: int, X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(
        isinstance(n_components, int)
        and not isinstance(n_components, bool)
        and values.ndim == 2
        and 1 <= n_components <= values.shape[1]
    )

def _lle_options_valid(
    eigen_solver: str,
    tol: float,
    max_iter: int,
    method: str,
    hessian_tol: float,
    modified_tol: float,
    random_state: int | None,
    n_jobs: None,
) -> bool:
    return bool(
        eigen_solver == "dense"
        and isinstance(tol, (int, float))
        and not isinstance(tol, bool)
        and np.isfinite(float(tol))
        and float(tol) >= 0.0
        and isinstance(max_iter, int)
        and not isinstance(max_iter, bool)
        and max_iter >= 1
        and method == "standard"
        and isinstance(hessian_tol, (int, float))
        and not isinstance(hessian_tol, bool)
        and np.isfinite(float(hessian_tol))
        and float(hessian_tol) >= 0.0
        and isinstance(modified_tol, (int, float))
        and not isinstance(modified_tol, bool)
        and np.isfinite(float(modified_tol))
        and float(modified_tol) >= 0.0
        and _random_state_valid(random_state)
        and n_jobs is None
    )

def _lle_fit_options_valid(
    eigen_solver: str,
    tol: float,
    max_iter: int,
    method: str,
    hessian_tol: float,
    modified_tol: float,
    neighbors_algorithm: str,
    random_state: int | None,
    n_jobs: None,
) -> bool:
    return bool(
        neighbors_algorithm in {"auto", "brute", "kd_tree", "ball_tree"}
        and _lle_options_valid(
            eigen_solver,
            tol,
            max_iter,
            method,
            hessian_tol,
            modified_tol,
            random_state,
            n_jobs,
        )
    )

def _lle_weights_valid(result: NDArray[np.float64], indices: NDArray[np.int64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    idx = np.asarray(indices)
    return bool(
        values.shape == idx.shape
        and np.all(np.isfinite(values))
        and np.allclose(np.sum(values, axis=1), 1.0)
    )

def _lle_graph_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_samples = np.asarray(X).shape[0]
    return bool(
        values.shape == (n_samples, n_samples)
        and np.all(np.isfinite(values))
        and np.allclose(np.sum(values, axis=1), 1.0)
    )

def _lle_matrix_valid(result: NDArray[np.float64], weights: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(weights)
    return bool(
        values.shape == source.shape
        and values.shape[0] == values.shape[1]
        and np.all(np.isfinite(values))
        and np.allclose(values, values.T)
    )

def _lle_state_valid(state: LocallyLinearEmbeddingState) -> bool:
    n_samples = state.training_data.shape[0]
    return bool(
        state.embedding.shape == (n_samples, state.n_components)
        and state.weights.shape == (n_samples, n_samples)
        and state.reconstruction_matrix.shape == (n_samples, n_samples)
        and state.training_data.shape[1] == state.n_features_in
        and 1 <= state.n_neighbors < n_samples
        and 1 <= state.n_components <= state.n_features_in
        and state.reg >= 0.0
        and state.eigen_solver == "dense"
        and state.method == "standard"
        and np.all(np.isfinite(state.embedding))
        and np.all(np.isfinite(state.training_data))
        and np.all(np.isfinite(state.weights))
        and np.all(np.isfinite(state.reconstruction_matrix))
        and np.isfinite(state.reconstruction_error)
        and state.reconstruction_error >= 0.0
        and np.allclose(np.sum(state.weights, axis=1), 1.0)
        and np.allclose(state.reconstruction_matrix, state.reconstruction_matrix.T)
    )

def _lle_feature_count_matches(X: NDArray[np.float64], state: LocallyLinearEmbeddingState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in)

def _lle_transform_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: LocallyLinearEmbeddingState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (np.asarray(X).shape[0], state.n_components) and np.all(np.isfinite(values)))

@register_atom(witness_classical_mds_dissimilarity_matrix)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda metric: _metric_valid(metric), "metric must be euclidean or precomputed")
@icontract.require(lambda metric_params: _metric_params_valid(metric_params), "metric_params are outside this atom scope")
@icontract.require(lambda X, metric: _precomputed_shape_valid(X, metric), "precomputed dissimilarities must be square")
@icontract.ensure(lambda result, X: _dissimilarity_valid(result, X), "dissimilarities must be finite symmetric sample-by-sample distances")
def classical_mds_dissimilarity_matrix(
    X: NDArray[np.float64],
    *,
    metric: str = "euclidean",
    metric_params: None = None,
) -> NDArray[np.float64]:
    from sklearn.metrics import pairwise_distances
    from sklearn.utils import check_symmetric
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Compute the dense dissimilarity matrix used by classical MDS."""
    del metric_params
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    if metric == "precomputed":
        return np.asarray(check_symmetric(checked_x, raise_exception=True), dtype=np.float64)
    return np.asarray(pairwise_distances(checked_x, metric="euclidean"), dtype=np.float64)

@register_atom(witness_classical_mds_double_center)
@icontract.require(lambda dissimilarity_matrix: _square_matrix(dissimilarity_matrix), "dissimilarity_matrix must be square")
@icontract.require(lambda dissimilarity_matrix: _finite_matrix(dissimilarity_matrix), "dissimilarity_matrix must contain only finite values")
@icontract.ensure(lambda result, dissimilarity_matrix: _centered_matrix_valid(result, dissimilarity_matrix), "double-centered matrix must be finite and mean centered")
def classical_mds_double_center(dissimilarity_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Center squared distances so each row and column has mean zero."""
    checked = check_array(dissimilarity_matrix, dtype=np.float64, ensure_2d=True)
    B = checked**2
    B = B.astype(np.float64)
    B -= np.mean(B, axis=0)
    B -= np.mean(B, axis=1, keepdims=True)
    B *= -0.5
    return B

@register_atom(witness_classical_mds_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda n_components, X: _n_components_valid(n_components, X), "n_components must be between one and sample count")
@icontract.require(lambda metric: _metric_valid(metric), "metric must be euclidean or precomputed")
@icontract.require(lambda metric_params: _metric_params_valid(metric_params), "metric_params are outside this atom scope")
@icontract.require(lambda X, metric: _precomputed_shape_valid(X, metric), "precomputed dissimilarities must be square")
@icontract.ensure(lambda result: _classical_mds_state_valid(result), "Classical MDS state must contain finite embedding coordinates")
def classical_mds_fit(
    X: NDArray[np.float64],
    *,
    n_components: int = 2,
    metric: str = "euclidean",
    metric_params: None = None,
) -> ClassicalMDSState:
    from sklearn.utils.extmath import _deterministic_vector_sign_flip, svd_flip
    """Fit low-dimensional coordinates from centered pairwise distances."""
    dissimilarity_matrix = classical_mds_dissimilarity_matrix(X, metric=metric, metric_params=metric_params)
    centered = classical_mds_double_center(dissimilarity_matrix)

    eigenvalues, eigenvectors = linalg.eigh(centered)
    eigenvalues = eigenvalues[::-1][:n_components]
    eigenvectors = eigenvectors[:, ::-1][:, :n_components]
    eigenvectors, _ = svd_flip(eigenvectors, None)
    embedding = np.sqrt(eigenvalues) * eigenvectors

    return ClassicalMDSState(
        embedding=np.asarray(embedding, dtype=np.float64),
        dissimilarity_matrix=dissimilarity_matrix,
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        n_components=n_components,
        metric=metric,
        n_features_in=int(np.asarray(X).shape[1]),
    )

def _metric_smacof_single(
    dissimilarities: NDArray[np.float64],
    *,
    n_components: int,
    init: NDArray[np.float64] | None,
    max_iter: int,
    eps: float,
    random_state: int | None,
    normalized_stress: bool,
) -> SMACOFState:
    from sklearn.metrics import pairwise_distances
    from sklearn.utils import check_symmetric
    checked = np.asarray(check_symmetric(dissimilarities, raise_exception=True), dtype=np.float64)
    n_samples = checked.shape[0]
    if init is None:
        rng = np.random.RandomState(random_state)
        embedding = rng.uniform(size=n_samples * n_components).reshape((n_samples, n_components))
    else:
        embedding = np.asarray(init, dtype=np.float64).copy()

    distances = pairwise_distances(embedding, metric="euclidean")
    old_stress: float | None = None
    n_iter = 0
    stress = 0.0
    for iteration in range(max_iter):
        disparities = checked
        distances[distances == 0.0] = 1e-5
        ratio = disparities / distances
        B = -ratio
        B[np.arange(n_samples), np.arange(n_samples)] += ratio.sum(axis=1)
        embedding = (B @ embedding) / n_samples
        distances = pairwise_distances(embedding, metric="euclidean")
        stress = float(np.sum((distances.ravel() - disparities.ravel()) ** 2) / 2.0)
        n_iter = iteration + 1
        if old_stress is not None:
            sum_squared_distances = float(np.sum(distances.ravel() ** 2))
            if ((old_stress - stress) / (sum_squared_distances / 2.0)) < eps:
                break
        old_stress = stress

    if normalized_stress:
        sum_squared_distances = float(np.sum(distances.ravel() ** 2))
        stress = float(np.sqrt(stress / (sum_squared_distances / 2.0)))

    return SMACOFState(
        embedding=np.asarray(embedding, dtype=np.float64),
        stress=float(stress),
        n_iter=int(n_iter),
        dissimilarity_matrix=checked,
        n_components=int(n_components),
        normalized_stress=normalized_stress,
    )

@register_atom(witness_smacof)
@icontract.require(lambda dissimilarities: _square_matrix(dissimilarities), "dissimilarities must be square")
@icontract.require(lambda dissimilarities: _finite_matrix(dissimilarities), "dissimilarities must contain only finite values")
@icontract.require(lambda metric: metric is True, "only metric SMACOF is covered")
@icontract.require(lambda n_components, dissimilarities: _n_components_valid(n_components, dissimilarities), "n_components must be between one and sample count")
@icontract.require(lambda init, dissimilarities, n_components: _init_shape_valid(init, np.asarray(dissimilarities).shape[0], n_components), "init must match sample and component counts")
@icontract.require(lambda n_init: n_init == 1, "only n_init=1 is covered")
@icontract.require(lambda n_jobs: n_jobs is None, "parallel SMACOF is outside this atom scope")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda verbose: verbose in {False, 0}, "verbose output is outside this atom scope")
@icontract.require(lambda eps: _positive_finite(eps), "eps must be positive")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be an integer or None")
@icontract.require(lambda return_n_iter: isinstance(return_n_iter, bool), "return_n_iter must be boolean")
@icontract.require(lambda normalized_stress: _normalized_stress_valid(normalized_stress), "normalized_stress must be boolean or auto")
@icontract.ensure(lambda result: _smacof_state_valid(result), "SMACOF state must contain finite embedding and stress")
def smacof(
    dissimilarities: NDArray[np.float64],
    *,
    metric: bool = True,
    n_components: int = 2,
    init: NDArray[np.float64] | None = None,
    n_init: int = 1,
    n_jobs: None = None,
    max_iter: int = 300,
    verbose: int | bool = 0,
    eps: float = 1e-6,
    random_state: int | None = None,
    return_n_iter: bool = False,
    normalized_stress: bool | str = "auto",
) -> SMACOFState:
    from sklearn.utils import check_symmetric
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Fit a dense metric SMACOF embedding from dissimilarities."""
    del metric, n_init, n_jobs, verbose, return_n_iter
    checked = check_array(dissimilarities, dtype=np.float64, ensure_2d=True)
    checked = np.asarray(check_symmetric(checked, raise_exception=True), dtype=np.float64)
    norm_stress = False if normalized_stress == "auto" else bool(normalized_stress)
    return _metric_smacof_single(
        checked,
        n_components=n_components,
        init=init,
        max_iter=max_iter,
        eps=eps,
        random_state=random_state,
        normalized_stress=norm_stress,
    )

@register_atom(witness_mds_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda n_components, X: _n_components_valid(n_components, X), "n_components must be between one and sample count")
@icontract.require(lambda metric_mds: metric_mds is True, "only metric MDS is covered")
@icontract.require(lambda n_init: n_init == 1, "only n_init=1 is covered")
@icontract.require(lambda init, X, n_components: _init_shape_valid(init, np.asarray(X).shape[0], n_components), "init must match sample and component counts")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda verbose: verbose in {False, 0}, "verbose output is outside this atom scope")
@icontract.require(lambda eps: _positive_finite(eps), "eps must be positive")
@icontract.require(lambda n_jobs: n_jobs is None, "parallel MDS is outside this atom scope")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be an integer or None")
@icontract.require(lambda metric: _metric_valid(metric), "metric must be euclidean or precomputed")
@icontract.require(lambda metric_params: _metric_params_valid(metric_params), "metric_params are outside this atom scope")
@icontract.require(lambda X, metric: _precomputed_shape_valid(X, metric), "precomputed dissimilarities must be square")
@icontract.require(lambda normalized_stress: _normalized_stress_valid(normalized_stress), "normalized_stress must be boolean or auto")
@icontract.ensure(lambda result: _mds_state_valid(result), "MDS state must contain finite embedding and stress")
def mds_fit(
    X: NDArray[np.float64],
    *,
    n_components: int = 2,
    metric_mds: bool = True,
    n_init: int = 1,
    init: NDArray[np.float64] | None = None,
    max_iter: int = 300,
    verbose: int | bool = 0,
    eps: float = 1e-6,
    n_jobs: None = None,
    random_state: int | None = None,
    metric: str = "euclidean",
    metric_params: None = None,
    normalized_stress: bool | str = "auto",
) -> MDSState:
    """Fit dense metric MDS coordinates with one SMACOF initialization."""
    del metric_mds, n_init, verbose, n_jobs, metric_params
    dissimilarity_matrix = classical_mds_dissimilarity_matrix(X, metric=metric, metric_params=None)
    smacof_state = smacof(
        dissimilarity_matrix,
        n_components=n_components,
        init=init,
        max_iter=max_iter,
        eps=eps,
        random_state=random_state,
        normalized_stress=normalized_stress,
    )
    n_features_in = int(np.asarray(X).shape[1])
    return MDSState(
        embedding=smacof_state.embedding,
        stress=smacof_state.stress,
        n_iter=smacof_state.n_iter,
        dissimilarity_matrix=dissimilarity_matrix,
        n_components=n_components,
        metric=metric,
        metric_mds=True,
        normalized_stress=smacof_state.normalized_stress,
        n_features_in=n_features_in,
    )

def _spectral_set_diag(laplacian: NDArray[np.float64], value: float, norm_laplacian: bool) -> NDArray[np.float64]:
    result = np.asarray(laplacian, dtype=np.float64).copy()
    if norm_laplacian:
        result.flat[:: result.shape[0] + 1] = value
    return result

@register_atom(witness_spectral_embedding)
@icontract.require(lambda adjacency: _square_matrix(adjacency), "adjacency must be square")
@icontract.require(lambda adjacency: _finite_matrix(adjacency), "adjacency must contain only finite values")
@icontract.require(lambda n_components, adjacency: _spectral_components_valid(n_components, adjacency), "n_components must be positive and below sample count")
@icontract.require(lambda eigen_solver: _eigen_solver_valid(eigen_solver), "only arpack/default eigen solving is covered")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be an integer or None")
@icontract.require(lambda eigen_tol: _eigen_tol_valid(eigen_tol), "eigen_tol must be non-negative or auto")
@icontract.require(lambda norm_laplacian: isinstance(norm_laplacian, bool), "norm_laplacian must be boolean")
@icontract.require(lambda drop_first: isinstance(drop_first, bool), "drop_first must be boolean")
@icontract.ensure(lambda result, adjacency, n_components: _spectral_embedding_valid(result, adjacency, n_components), "spectral embedding must contain finite coordinates")
def spectral_embedding(
    adjacency: NDArray[np.float64],
    *,
    n_components: int = 8,
    eigen_solver: str | None = None,
    random_state: int | None = None,
    eigen_tol: float | str = "auto",
    norm_laplacian: bool = True,
    drop_first: bool = True,
) -> NDArray[np.float64]:
    from sklearn.utils import check_symmetric
    from sklearn.utils._arpack import _init_arpack_v0
    from sklearn.utils.extmath import _deterministic_vector_sign_flip, svd_flip
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Project a dense affinity graph onto Laplacian eigenvectors."""
    del eigen_solver
    checked = check_array(adjacency, dtype=np.float64, ensure_2d=True)
    checked = np.asarray(check_symmetric(checked, raise_exception=True), dtype=np.float64)
    effective_components = n_components + 1 if drop_first else n_components
    laplacian, diagonal = csgraph_laplacian(checked, normed=norm_laplacian, return_diag=True)
    laplacian = _spectral_set_diag(np.asarray(laplacian, dtype=np.float64), 1.0, norm_laplacian)
    laplacian *= -1.0
    rng = np.random.RandomState(random_state)
    tolerance = 0.0 if eigen_tol == "auto" else float(eigen_tol)
    v0 = _init_arpack_v0(laplacian.shape[0], rng)
    _, diffusion_map = eigsh(laplacian, k=effective_components, sigma=1.0, which="LM", tol=tolerance, v0=v0)
    embedding = diffusion_map.T[effective_components::-1]
    if norm_laplacian:
        embedding = embedding / diagonal
    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return np.asarray(embedding[1:effective_components].T, dtype=np.float64)
    return np.asarray(embedding[:effective_components].T, dtype=np.float64)

@register_atom(witness_spectral_embedding_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda n_components, X: _spectral_components_valid(n_components, X), "n_components must be positive and below sample count")
@icontract.require(lambda affinity: _spectral_affinity_valid(affinity), "only rbf and precomputed affinities are covered")
@icontract.require(lambda gamma: _spectral_gamma_valid(gamma), "gamma must be positive when provided")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be an integer or None")
@icontract.require(lambda eigen_solver: _eigen_solver_valid(eigen_solver), "only arpack/default eigen solving is covered")
@icontract.require(lambda eigen_tol: _eigen_tol_valid(eigen_tol), "eigen_tol must be non-negative or auto")
@icontract.require(lambda n_neighbors: _spectral_optional_none(n_neighbors), "nearest-neighbor affinity is outside this atom scope")
@icontract.require(lambda n_jobs: _spectral_optional_none(n_jobs), "parallel neighbor construction is outside this atom scope")
@icontract.require(lambda X, affinity: _precomputed_shape_valid(X, affinity), "precomputed affinity must be square")
@icontract.ensure(lambda result: _spectral_state_valid(result), "Spectral embedding state must contain finite coordinates")
def spectral_embedding_fit(
    X: NDArray[np.float64],
    *,
    n_components: int = 2,
    affinity: str = "rbf",
    gamma: float | None = None,
    random_state: int | None = None,
    eigen_solver: str | None = None,
    eigen_tol: float | str = "auto",
    n_neighbors: None = None,
    n_jobs: None = None,
) -> SpectralEmbeddingState:
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.utils import check_symmetric
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Fit dense spectral embedding coordinates from rbf or precomputed affinity."""
    del n_neighbors, n_jobs
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    if affinity == "precomputed":
        affinity_matrix = np.asarray(check_symmetric(checked, raise_exception=True), dtype=np.float64)
        n_features_in = int(checked.shape[1])
        gamma_value = gamma
    else:
        gamma_value = float(gamma) if gamma is not None else 1.0 / checked.shape[1]
        affinity_matrix = np.asarray(rbf_kernel(checked, gamma=gamma_value), dtype=np.float64)
        n_features_in = int(checked.shape[1])
    embedding = spectral_embedding(
        affinity_matrix,
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        norm_laplacian=True,
        drop_first=True,
    )
    return SpectralEmbeddingState(
        embedding=embedding,
        affinity_matrix=affinity_matrix,
        n_components=int(n_components),
        affinity=affinity,
        gamma=gamma_value,
        eigen_solver="arpack",
        n_features_in=n_features_in,
    )

def _center_precomputed_kernel(kernel_matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    checked = np.asarray(kernel_matrix, dtype=np.float64)
    n_samples = checked.shape[0]
    rows = np.sum(checked, axis=0) / n_samples
    all_mean = float(np.sum(rows) / n_samples)
    centered = checked.copy()
    column_means = (np.sum(centered, axis=1) / n_samples)[:, np.newaxis]
    centered -= rows
    centered -= column_means
    centered += all_mean
    return np.asarray(centered, dtype=np.float64), np.asarray(rows, dtype=np.float64), all_mean

def _fit_precomputed_kernel_pca(
    kernel_matrix: NDArray[np.float64],
    n_components: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    from sklearn.utils.extmath import _deterministic_vector_sign_flip, svd_flip
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    centered, rows, all_mean = _center_precomputed_kernel(kernel_matrix)
    n_samples = centered.shape[0]
    resolved_components = min(n_samples, int(n_components))
    eigenvalues, eigenvectors = linalg.eigh(
        centered,
        subset_by_index=(n_samples - resolved_components, n_samples - 1),
    )
    eigenvalues = _check_psd_eigenvalues(eigenvalues, enable_warnings=False)
    eigenvectors, _ = svd_flip(u=eigenvectors, v=None)
    indices = eigenvalues.argsort()[::-1]
    ordered_values = np.asarray(eigenvalues[indices], dtype=np.float64)
    ordered_vectors = np.asarray(eigenvectors[:, indices], dtype=np.float64)
    embedding = np.asarray(ordered_vectors * np.sqrt(ordered_values), dtype=np.float64)
    return embedding, ordered_values, ordered_vectors, rows, all_mean

@register_atom(witness_isomap_neighbors_graph)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda n_neighbors, X: _isomap_neighbors_valid(n_neighbors, X), "n_neighbors must be positive and below sample count")
@icontract.require(
    lambda radius, neighbors_algorithm, n_jobs, metric, p, metric_params: _isomap_neighbor_options_valid(
        radius,
        neighbors_algorithm,
        n_jobs,
        metric,
        p,
        metric_params,
    ),
    "only dense n-neighbor minkowski graph construction is covered",
)
@icontract.ensure(lambda result, X: _isomap_graph_valid(result, X), "neighbor graph must contain finite nonnegative distances")
def isomap_neighbors_graph(
    X: NDArray[np.float64],
    *,
    n_neighbors: int = 5,
    radius: None = None,
    neighbors_algorithm: str = "auto",
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
    n_jobs: None = None,
) -> NDArray[np.float64]:
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Build the dense distance graph used by the n-neighbor Isomap path."""
    del radius
    checked = check_array(X, dtype=np.float64, ensure_2d=True, ensure_min_samples=2)
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=neighbors_algorithm,
        metric=metric,
        p=p,
        metric_params=metric_params,
        n_jobs=n_jobs,
    )
    nbrs.fit(checked)
    graph = kneighbors_graph(
        nbrs,
        n_neighbors,
        mode="distance",
        metric=metric,
        p=p,
        metric_params=metric_params,
        n_jobs=n_jobs,
    )
    return np.asarray(graph.toarray(), dtype=np.float64)

@register_atom(witness_isomap_geodesic_distances)
@icontract.require(lambda neighbors_graph: _square_matrix(neighbors_graph), "neighbors_graph must be square")
@icontract.require(lambda neighbors_graph: _finite_matrix(neighbors_graph), "neighbors_graph must contain finite values")
@icontract.require(lambda path_method: path_method in {"auto", "FW", "D"}, "path_method must be auto, FW, or D")
@icontract.ensure(lambda result, neighbors_graph: _isomap_distances_valid(result, neighbors_graph), "geodesic distances must be finite and symmetric")
def isomap_geodesic_distances(
    neighbors_graph: NDArray[np.float64],
    *,
    path_method: str = "auto",
) -> NDArray[np.float64]:
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Compute all-pairs geodesic distances from an Isomap neighbor graph."""
    graph = check_array(neighbors_graph, dtype=np.float64, ensure_2d=True)
    n_components, _ = connected_components(graph)
    if n_components != 1:
        raise ValueError("Isomap neighbor graph must be connected in this atom scope")
    return np.asarray(shortest_path(graph, method=path_method, directed=False), dtype=np.float64)

@register_atom(witness_isomap_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda n_neighbors, X: _isomap_neighbors_valid(n_neighbors, X), "n_neighbors must be positive and below sample count")
@icontract.require(lambda n_components, X: _isomap_components_valid(n_components, X), "n_components must be positive and no larger than sample count")
@icontract.require(
    lambda radius, eigen_solver, tol, max_iter, path_method, neighbors_algorithm, n_jobs, metric, p, metric_params: _isomap_options_valid(
        radius,
        eigen_solver,
        tol,
        max_iter,
        path_method,
        neighbors_algorithm,
        n_jobs,
        metric,
        p,
        metric_params,
    ),
    "only dense n-neighbor minkowski Isomap with dense KernelPCA is covered",
)
@icontract.ensure(lambda result: _isomap_state_valid(result), "Isomap state must contain finite embedding and geodesic-kernel data")
def isomap_fit(
    X: NDArray[np.float64],
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
) -> IsomapState:
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Fit dense n-neighbor Isomap coordinates and reusable transform state."""
    del radius, eigen_solver, tol, max_iter
    checked = check_array(X, dtype=np.float64, ensure_2d=True, ensure_min_samples=2)
    graph = isomap_neighbors_graph(
        checked,
        n_neighbors=n_neighbors,
        neighbors_algorithm=neighbors_algorithm,
        metric=metric,
        p=p,
        metric_params=metric_params,
        n_jobs=n_jobs,
    )
    dist_matrix = isomap_geodesic_distances(graph, path_method=path_method)
    kernel_matrix = np.asarray(dist_matrix**2, dtype=np.float64)
    kernel_matrix *= -0.5
    embedding, eigenvalues, eigenvectors, centerer_rows, centerer_all = _fit_precomputed_kernel_pca(
        kernel_matrix,
        n_components,
    )
    return IsomapState(
        embedding=embedding,
        dist_matrix=np.asarray(dist_matrix, dtype=np.float64).copy(),
        training_data=np.asarray(checked, dtype=np.float64).copy(),
        eigenvalues=eigenvalues.copy(),
        eigenvectors=eigenvectors.copy(),
        kernel_centerer_rows=centerer_rows.copy(),
        kernel_centerer_all=centerer_all,
        n_neighbors=int(n_neighbors),
        n_components=int(min(checked.shape[0], n_components)),
        path_method=path_method,
        metric=metric,
        p=float(p),
        n_features_in=int(checked.shape[1]),
    )

@register_atom(witness_isomap_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda state: _isomap_state_valid(state), "state must be a fitted dense Isomap state")
@icontract.require(lambda X, state: _isomap_feature_count_matches(X, state), "X feature count must match fitted Isomap state")
@icontract.ensure(lambda result, X, state: _isomap_transform_valid(result, X, state), "Isomap transform must contain finite coordinates")
def isomap_transform(X: NDArray[np.float64], state: IsomapState) -> NDArray[np.float64]:
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Project query samples through a fitted dense Isomap state."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    nbrs = NearestNeighbors(
        n_neighbors=state.n_neighbors,
        algorithm="auto",
        metric=state.metric,
        p=state.p,
        metric_params=None,
        n_jobs=None,
    )
    nbrs.fit(state.training_data)
    distances, indices = nbrs.kneighbors(checked, return_distance=True)
    n_queries = checked.shape[0]
    n_samples_fit = state.training_data.shape[0]
    query_kernel = np.zeros((n_queries, n_samples_fit), dtype=np.float64)
    for row in range(n_queries):
        query_kernel[row] = np.min(state.dist_matrix[indices[row]] + distances[row][:, np.newaxis], axis=0)
    query_kernel **= 2
    query_kernel *= -0.5
    predicted_column_means = (np.sum(query_kernel, axis=1) / n_samples_fit)[:, np.newaxis]
    query_kernel -= state.kernel_centerer_rows
    query_kernel -= predicted_column_means
    query_kernel += state.kernel_centerer_all
    non_zero_indices = np.flatnonzero(state.eigenvalues)
    scaled_eigenvectors = np.zeros_like(state.eigenvectors)
    scaled_eigenvectors[:, non_zero_indices] = (
        state.eigenvectors[:, non_zero_indices] / np.sqrt(state.eigenvalues[non_zero_indices])
    )
    return np.asarray(np.dot(query_kernel, scaled_eigenvectors), dtype=np.float64)

@register_atom(witness_isomap_reconstruction_error)
@icontract.require(lambda state: _isomap_state_valid(state), "state must be a fitted dense Isomap state")
@icontract.ensure(lambda result: _nonnegative_finite_scalar(result), "reconstruction error must be finite and nonnegative")
def isomap_reconstruction_error(state: IsomapState) -> float:
    """Compute Isomap reconstruction error from the centered geodesic kernel."""
    kernel_matrix = -0.5 * state.dist_matrix**2
    centered_kernel, _, _ = _center_precomputed_kernel(kernel_matrix)
    residual = float(np.sum(centered_kernel**2) - np.sum(state.eigenvalues**2))
    return float(np.sqrt(max(residual, 0.0)) / state.dist_matrix.shape[0])

@register_atom(witness_lle_barycenter_weights)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda Y: _matrix_2d(Y), "Y must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda Y: _finite_matrix(Y), "Y must contain only finite values")
@icontract.require(lambda indices: _integer_index_matrix(indices), "indices must be a 2D integer matrix")
@icontract.require(lambda X, Y, indices: _lle_indices_valid(X, Y, indices), "indices must select rows of Y for each X row")
@icontract.require(lambda reg: _lle_reg_valid(reg), "reg must be nonnegative and finite")
@icontract.ensure(lambda result, indices: _lle_weights_valid(result, indices), "barycenter weights must be finite and row-normalized")
def lle_barycenter_weights(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    indices: NDArray[np.int64],
    *,
    reg: float = 1e-3,
) -> NDArray[np.float64]:
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Compute row-normalized local reconstruction weights for LLE."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(Y, dtype=np.float64, ensure_2d=True)
    checked_indices = check_array(indices, dtype=np.int64, ensure_2d=True)
    n_samples, n_neighbors = checked_indices.shape
    weights = np.empty((n_samples, n_neighbors), dtype=np.float64)
    ones = np.ones(n_neighbors, dtype=np.float64)
    for row, neighbor_indices in enumerate(checked_indices):
        centered = checked_y[neighbor_indices] - checked_x[row]
        gram = np.dot(centered, centered.T)
        trace = float(np.trace(gram))
        regularization = float(reg) * trace if trace > 0.0 else float(reg)
        gram.flat[:: n_neighbors + 1] += regularization
        raw_weights = linalg.solve(gram, ones, assume_a="pos")
        weights[row, :] = raw_weights / np.sum(raw_weights)
    return np.asarray(weights, dtype=np.float64)

@register_atom(witness_lle_barycenter_graph)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda n_neighbors, X: _lle_neighbors_valid(n_neighbors, X), "n_neighbors must be positive and below sample count")
@icontract.require(lambda reg: _lle_reg_valid(reg), "reg must be nonnegative and finite")
@icontract.require(lambda n_jobs: n_jobs is None, "parallel neighbor search is outside this atom scope")
@icontract.ensure(lambda result, X: _lle_graph_valid(result, X), "LLE graph must contain finite row-normalized weights")
def lle_barycenter_graph(
    X: NDArray[np.float64],
    *,
    n_neighbors: int,
    reg: float = 1e-3,
    n_jobs: None = None,
) -> NDArray[np.float64]:
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Build the dense barycenter-weight matrix for standard LLE."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True, ensure_min_samples=2)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(checked)
    indices = nbrs.kneighbors(checked, return_distance=False)[:, 1:]
    local_weights = lle_barycenter_weights(checked, checked, np.asarray(indices, dtype=np.int64), reg=reg)
    graph = np.zeros((checked.shape[0], checked.shape[0]), dtype=np.float64)
    for row, neighbor_indices in enumerate(indices):
        graph[row, neighbor_indices] = local_weights[row]
    return graph

@register_atom(witness_lle_standard_reconstruction_matrix)
@icontract.require(lambda weights: _square_matrix(weights), "weights must be square")
@icontract.require(lambda weights: _finite_matrix(weights), "weights must contain only finite values")
@icontract.ensure(lambda result, weights: _lle_matrix_valid(result, weights), "reconstruction matrix must be finite and symmetric")
def lle_standard_reconstruction_matrix(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Compute the dense standard LLE matrix (I - W)'(I - W)."""
    checked = check_array(weights, dtype=np.float64, ensure_2d=True)
    matrix = np.asarray(checked.T @ checked - checked.T - checked, dtype=np.float64)
    matrix.flat[:: matrix.shape[0] + 1] += 1.0
    return matrix

@register_atom(witness_locally_linear_embedding)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda n_neighbors, X: _lle_neighbors_valid(n_neighbors, X), "n_neighbors must be positive and below sample count")
@icontract.require(lambda n_components, X: _lle_components_valid(n_components, X), "n_components must be positive and no larger than feature count")
@icontract.require(lambda reg: _lle_reg_valid(reg), "reg must be nonnegative and finite")
@icontract.require(
    lambda eigen_solver, tol, max_iter, method, hessian_tol, modified_tol, random_state, n_jobs: _lle_options_valid(
        eigen_solver,
        tol,
        max_iter,
        method,
        hessian_tol,
        modified_tol,
        random_state,
        n_jobs,
    ),
    "only standard dense LLE is covered",
)
@icontract.ensure(lambda result: _lle_state_valid(result), "LLE state must contain finite dense standard embedding data")
def locally_linear_embedding(
    X: NDArray[np.float64],
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
) -> LocallyLinearEmbeddingState:
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Fit standard dense locally linear embedding coordinates."""
    del eigen_solver, tol, max_iter, method, hessian_tol, modified_tol, random_state
    checked = check_array(X, dtype=np.float64, ensure_2d=True, ensure_min_samples=2)
    weights = lle_barycenter_graph(checked, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)
    reconstruction_matrix = lle_standard_reconstruction_matrix(weights)
    eigen_values, eigen_vectors = linalg.eigh(
        reconstruction_matrix,
        subset_by_index=(1, n_components),
        overwrite_a=True,
    )
    order = np.argsort(np.abs(eigen_values))
    embedding = np.asarray(eigen_vectors[:, order], dtype=np.float64)
    reconstruction_error = float(np.sum(eigen_values))
    return LocallyLinearEmbeddingState(
        embedding=embedding,
        reconstruction_error=reconstruction_error,
        training_data=np.asarray(checked, dtype=np.float64).copy(),
        weights=weights.copy(),
        reconstruction_matrix=reconstruction_matrix.copy(),
        n_neighbors=int(n_neighbors),
        n_components=int(n_components),
        reg=float(reg),
        eigen_solver="dense",
        method="standard",
        n_features_in=int(checked.shape[1]),
    )

@register_atom(witness_locally_linear_embedding_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda n_neighbors, X: _lle_neighbors_valid(n_neighbors, X), "n_neighbors must be positive and below sample count")
@icontract.require(lambda n_components, X: _lle_components_valid(n_components, X), "n_components must be positive and no larger than feature count")
@icontract.require(lambda reg: _lle_reg_valid(reg), "reg must be nonnegative and finite")
@icontract.require(
    lambda eigen_solver, tol, max_iter, method, hessian_tol, modified_tol, neighbors_algorithm, random_state, n_jobs: _lle_fit_options_valid(
        eigen_solver,
        tol,
        max_iter,
        method,
        hessian_tol,
        modified_tol,
        neighbors_algorithm,
        random_state,
        n_jobs,
    ),
    "only standard dense LLE estimator fitting is covered",
)
@icontract.ensure(lambda result: _lle_state_valid(result), "LLE estimator state must contain finite dense standard embedding data")
def locally_linear_embedding_fit(
    X: NDArray[np.float64],
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
) -> LocallyLinearEmbeddingState:
    """Fit standard dense LocallyLinearEmbedding estimator state."""
    del neighbors_algorithm
    return locally_linear_embedding(
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

@register_atom(witness_locally_linear_embedding_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda state: _lle_state_valid(state), "state must be a fitted standard dense LLE state")
@icontract.require(lambda X, state: _lle_feature_count_matches(X, state), "X feature count must match fitted LLE state")
@icontract.ensure(lambda result, X, state: _lle_transform_valid(result, X, state), "LLE transform must contain finite coordinates")
def locally_linear_embedding_transform(
    X: NDArray[np.float64],
    state: LocallyLinearEmbeddingState,
) -> NDArray[np.float64]:
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph
    from sklearn.utils.validation import _check_psd_eigenvalues, check_array
    """Transform samples with a fitted standard dense LLE state."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    nbrs = NearestNeighbors(n_neighbors=state.n_neighbors)
    nbrs.fit(state.training_data)
    indices = nbrs.kneighbors(checked, n_neighbors=state.n_neighbors, return_distance=False)
    weights = lle_barycenter_weights(checked, state.training_data, np.asarray(indices, dtype=np.int64), reg=state.reg)
    transformed = np.empty((checked.shape[0], state.n_components), dtype=np.float64)
    for row in range(checked.shape[0]):
        transformed[row] = np.dot(state.embedding[indices[row]].T, weights[row])
    return transformed
