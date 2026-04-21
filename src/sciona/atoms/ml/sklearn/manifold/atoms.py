"""Manifold-learning atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_symmetric
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import check_array

from sciona.ghost.registry import register_atom

from .state_models import ClassicalMDSState, MDSState, SMACOFState
from .witnesses import (
    witness_classical_mds_dissimilarity_matrix,
    witness_classical_mds_double_center,
    witness_classical_mds_fit,
    witness_mds_fit,
    witness_smacof,
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
