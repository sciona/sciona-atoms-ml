"""Selected clustering atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings
from collections import defaultdict

import icontract
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import euclidean_distances, pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state, gen_batches
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import _check_sample_weight

from sciona.ghost.registry import register_atom

from .state_models import AffinityPropagationState, MeanShiftState
from .witnesses import (
    witness_affinity_propagation,
    witness_affinity_propagation_fit,
    witness_affinity_propagation_predict,
    witness_cluster_optics_dbscan,
    witness_cluster_optics_xi,
    witness_estimate_bandwidth,
    witness_kmeans_plusplus,
    witness_mean_shift,
    witness_mean_shift_fit,
    witness_mean_shift_predict,
)

MatrixLike = NDArray[np.float64] | sp.spmatrix | list[list[float]]
PreferenceLike = float | NDArray[np.float64] | list[float] | None
RandomStateLike = int | np.random.RandomState | None
AffinityPropagationResult = tuple[NDArray[np.int_], NDArray[np.int_]]
AffinityPropagationResultWithIter = tuple[NDArray[np.int_], NDArray[np.int_], int]
AffinityPropagationOutput = AffinityPropagationResult | AffinityPropagationResultWithIter
MeanShiftResult = tuple[NDArray[np.float64], NDArray[np.int_]]
KMeansPlusPlusResult = tuple[NDArray[np.float64], NDArray[np.int_]]
OpticsXiResult = tuple[NDArray[np.int_], NDArray[np.int_]]


def _is_2d_matrix(X: MatrixLike) -> bool:
    if sp.issparse(X):
        return bool(X.ndim == 2)
    return bool(np.asarray(X).ndim == 2)


def _sample_count(X: MatrixLike) -> int:
    return int(X.shape[0]) if sp.issparse(X) else int(np.asarray(X).shape[0])


def _feature_count(X: MatrixLike) -> int:
    return int(X.shape[1]) if sp.issparse(X) else int(np.asarray(X).shape[1])


def _is_square_matrix(X: MatrixLike) -> bool:
    if not _is_2d_matrix(X):
        return False
    shape = X.shape if sp.issparse(X) else np.asarray(X).shape
    return bool(shape[0] == shape[1])


def _damping_valid(damping: float) -> bool:
    return bool(0.5 <= float(damping) < 1.0)


def _positive_int(value: int) -> bool:
    return isinstance(value, int) and value >= 1


def _nonnegative_int(value: int) -> bool:
    return isinstance(value, int) and value >= 0


def _positive_int_or_none(value: int | None) -> bool:
    return value is None or (isinstance(value, int) and value >= 1)


def _quantile_valid(quantile: float) -> bool:
    return bool(0.0 <= float(quantile) <= 1.0)


def _bandwidth_valid(bandwidth: float | None) -> bool:
    return bandwidth is None or float(bandwidth) > 0.0


def _n_jobs_valid(n_jobs: int | None) -> bool:
    return n_jobs is None or isinstance(n_jobs, int)


def _affinity_valid(affinity: str) -> bool:
    return affinity in {"euclidean", "precomputed"}


def _preference_valid(preference: PreferenceLike) -> bool:
    if preference is None:
        return True
    values = np.asarray(preference, dtype=np.float64)
    return bool(values.ndim <= 1 and np.all(np.isfinite(values)))


def _preference_matches_samples(preference: PreferenceLike, X: MatrixLike) -> bool:
    if preference is None:
        return True
    values = np.asarray(preference, dtype=np.float64)
    return bool(values.ndim == 0 or values.shape == (_sample_count(X),))


def _seeds_valid(seeds: MatrixLike | None) -> bool:
    return seeds is None or _is_2d_matrix(seeds)


def _seeds_match_features(seeds: MatrixLike | None, X: MatrixLike) -> bool:
    return seeds is None or _feature_count(seeds) == _feature_count(X)


def _clusters_within_samples(n_clusters: int, X: MatrixLike) -> bool:
    return _positive_int(n_clusters) and n_clusters <= _sample_count(X)


def _vector_matches_samples(vector: NDArray[np.float64] | list[float] | None, X: MatrixLike) -> bool:
    if vector is None:
        return True
    return bool(np.asarray(vector).ndim == 1 and np.asarray(vector).shape[0] == _sample_count(X))


def _sample_weight_valid(sample_weight: NDArray[np.float64] | list[float] | None, X: MatrixLike) -> bool:
    if sample_weight is None:
        return True
    values = np.asarray(sample_weight, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape[0] == _sample_count(X) and np.all(values >= 0.0) and values.sum() > 0.0)


def _is_1d_numeric(vector: NDArray[np.float64] | NDArray[np.int_]) -> bool:
    return bool(np.asarray(vector).ndim == 1)


def _same_length_1d(
    reachability: NDArray[np.float64],
    core_distances: NDArray[np.float64],
    ordering: NDArray[np.int_],
) -> bool:
    return bool(
        np.asarray(reachability).shape == np.asarray(core_distances).shape
        and np.asarray(reachability).shape == np.asarray(ordering).shape
    )


def _ordering_permutation(ordering: NDArray[np.int_]) -> bool:
    values = np.asarray(ordering)
    if values.ndim != 1 or values.dtype.kind not in {"i", "u"}:
        return False
    n_samples = values.shape[0]
    return bool(np.array_equal(np.sort(values), np.arange(n_samples)))


def _nonnegative_float(value: float) -> bool:
    return bool(float(value) >= 0.0)


def _xi_valid(xi: float) -> bool:
    return bool(0.0 <= float(xi) <= 1.0)


def _optics_size_valid(size: int | float | None, n_samples: int, *, allow_none: bool) -> bool:
    if size is None:
        return allow_none
    if isinstance(size, int):
        return 2 <= size <= n_samples
    if isinstance(size, float):
        return 0.0 <= size <= 1.0
    return False


def _equal_similarities_and_preferences(S: NDArray[np.float64], preference: NDArray[np.float64]) -> bool:
    return bool(np.all(S == S.flat[0]) and np.all(preference == preference.flat[0]))


def _prepare_preference(preference: PreferenceLike, affinity_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    if preference is None:
        return np.asarray(np.median(affinity_matrix), dtype=np.float64)
    values = np.asarray(preference, dtype=np.float64)
    if values.ndim > 1:
        raise ValueError("preference must be a scalar or a vector")
    if values.ndim == 1 and values.shape[0] != affinity_matrix.shape[0]:
        raise ValueError("preference length must equal the sample count")
    return values


def _stored_preference(preference: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if preference.ndim == 0:
        return float(preference)
    return np.asarray(preference, dtype=np.float64).copy()


def _indices_and_labels_valid(centers: NDArray[np.int_], labels: NDArray[np.int_], n_samples: int) -> bool:
    if centers.ndim != 1 or labels.shape != (n_samples,):
        return False
    if centers.size > 0 and (centers.min() < 0 or centers.max() >= n_samples):
        return False
    return bool(np.all((labels == -1) | ((labels >= 0) & (labels < max(centers.size, 1)))))


def _affinity_result_valid(result: AffinityPropagationOutput, S: MatrixLike, return_n_iter: bool) -> bool:
    expected_len = 3 if return_n_iter else 2
    if len(result) != expected_len:
        return False
    centers = np.asarray(result[0])
    labels = np.asarray(result[1])
    if centers.dtype.kind not in {"i", "u"} or labels.dtype.kind not in {"i", "u"}:
        return False
    if return_n_iter and (not isinstance(result[2], int) or result[2] < 0):
        return False
    return _indices_and_labels_valid(centers.astype(np.int_), labels.astype(np.int_), _sample_count(S))


def _state_valid(state: AffinityPropagationState) -> bool:
    n_samples = state.affinity_matrix.shape[0]
    centers_ok = _indices_and_labels_valid(state.cluster_centers_indices, state.labels, n_samples)
    centers_shape_ok = (
        state.cluster_centers is None
        if state.affinity == "precomputed"
        else state.cluster_centers is not None
        and state.cluster_centers.shape == (state.cluster_centers_indices.shape[0], state.n_features_in)
    )
    return bool(
        state.affinity_matrix.ndim == 2
        and state.affinity_matrix.shape[0] == state.affinity_matrix.shape[1]
        and state.n_iter >= 0
        and _affinity_valid(state.affinity)
        and _damping_valid(state.damping)
        and state.n_features_in >= 1
        and centers_ok
        and centers_shape_ok
    )


def _prediction_valid(result: NDArray[np.int_], X: MatrixLike) -> bool:
    labels = np.asarray(result)
    return bool(labels.shape == (_sample_count(X),) and labels.dtype.kind in {"i", "u"} and np.all(labels >= -1))


def _bandwidth_result_valid(result: float | np.float64) -> bool:
    return bool(np.isfinite(result) and float(result) >= 0.0)


def _mean_shift_result_valid(result: MeanShiftResult, X: MatrixLike) -> bool:
    centers, labels = result
    return bool(
        centers.ndim == 2
        and centers.shape[1] == _feature_count(X)
        and labels.shape == (_sample_count(X),)
        and np.all(np.isfinite(centers))
        and labels.dtype.kind in {"i", "u"}
        and np.all(labels >= -1)
    )


def _mean_shift_state_valid(state: MeanShiftState) -> bool:
    return bool(
        state.cluster_centers.ndim == 2
        and state.cluster_centers.shape[0] >= 1
        and state.cluster_centers.shape[1] == state.n_features_in
        and state.labels.ndim == 1
        and state.labels.size >= 1
        and np.all(np.isfinite(state.cluster_centers))
        and np.isfinite(state.bandwidth)
        and state.bandwidth >= 0.0
        and state.n_iter >= 0
        and state.n_features_in >= 1
        and state.labels.dtype.kind in {"i", "u"}
        and np.all(state.labels >= -1)
    )


def _kmeans_plusplus_result_valid(result: KMeansPlusPlusResult, X: MatrixLike, n_clusters: int) -> bool:
    centers, indices = result
    n_samples = _sample_count(X)
    return bool(
        centers.shape == (n_clusters, _feature_count(X))
        and indices.shape == (n_clusters,)
        and np.all(np.isfinite(centers))
        and indices.dtype.kind in {"i", "u"}
        and np.all(indices >= 0)
        and np.all(indices < n_samples)
    )


def _optics_labels_valid(result: NDArray[np.int_], reachability: NDArray[np.float64]) -> bool:
    labels = np.asarray(result)
    return bool(
        labels.shape == np.asarray(reachability).shape
        and labels.dtype.kind in {"i", "u"}
        and np.all(labels >= -1)
    )


def _optics_xi_result_valid(result: OpticsXiResult, reachability: NDArray[np.float64]) -> bool:
    labels, clusters = result
    n_samples = np.asarray(reachability).shape[0]
    clusters_ok = clusters.ndim == 2 and clusters.shape[1] == 2 and clusters.dtype.kind in {"i", "u"}
    if clusters.size:
        clusters_ok = bool(clusters_ok and np.all(clusters >= 0) and np.all(clusters[:, 0] <= clusters[:, 1]) and np.all(clusters < n_samples))
    return bool(_optics_labels_valid(labels, reachability) and clusters_ok)


def _as_dense_float_matrix(X: MatrixLike) -> NDArray[np.float64]:
    if sp.issparse(X):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _get_bin_seeds(X: NDArray[np.float64], bin_size: float, min_bin_freq: int = 1) -> NDArray[np.float64]:
    if bin_size == 0:
        return X

    bin_sizes: defaultdict[tuple[np.float64, ...], int] = defaultdict(int)
    for point in X:
        binned_point = np.round(point / bin_size)
        bin_sizes[tuple(binned_point)] += 1

    bin_seeds = np.array(
        [point for point, freq in bin_sizes.items() if freq >= min_bin_freq],
        dtype=np.float32,
    )
    if len(bin_seeds) == len(X):
        warnings.warn(
            "Binning data failed with provided bin_size=%f, using data points as seeds." % bin_size
        )
        return X
    return np.asarray(bin_seeds * bin_size, dtype=np.float64)


def _mean_shift_single_seed(
    my_mean: NDArray[np.float64],
    X: NDArray[np.float64],
    nbrs: NearestNeighbors,
    max_iter: int,
) -> tuple[tuple[float, ...], int, int]:
    bandwidth = float(nbrs.get_params()["radius"])
    stop_thresh = 1e-3 * bandwidth
    completed_iterations = 0
    while True:
        neighbor_indices = nbrs.radius_neighbors([my_mean], bandwidth, return_distance=False)[0]
        points_within = X[neighbor_indices]
        if len(points_within) == 0:
            break
        old_mean = my_mean
        my_mean = np.mean(points_within, axis=0)
        if np.linalg.norm(my_mean - old_mean) <= stop_thresh or completed_iterations == max_iter:
            break
        completed_iterations += 1
    return tuple(float(value) for value in my_mean), len(points_within), completed_iterations


def _mean_shift_fit_core(
    X: MatrixLike,
    *,
    bandwidth: float | None,
    seeds: MatrixLike | None,
    bin_seeding: bool,
    min_bin_freq: int,
    cluster_all: bool,
    max_iter: int,
    n_jobs: int | None,
) -> MeanShiftState:
    checked_x = np.asarray(check_array(X, dtype=[np.float64, np.float32]), dtype=np.float64)
    fitted_bandwidth = float(bandwidth) if bandwidth is not None else float(estimate_bandwidth(checked_x, n_jobs=n_jobs))

    if seeds is None:
        if bin_seeding:
            seed_points = _get_bin_seeds(checked_x, fitted_bandwidth, min_bin_freq)
        else:
            seed_points = checked_x
    else:
        seed_points = np.asarray(check_array(seeds, dtype=[np.float64, np.float32]), dtype=np.float64)

    n_samples, n_features = checked_x.shape
    center_intensity: dict[tuple[float, ...], int] = {}
    nbrs = NearestNeighbors(radius=fitted_bandwidth, n_jobs=1).fit(checked_x)

    all_results = Parallel(n_jobs=n_jobs)(
        delayed(_mean_shift_single_seed)(seed, checked_x, nbrs, max_iter)
        for seed in seed_points
    )
    for center, intensity, _ in all_results:
        if intensity:
            center_intensity[center] = intensity

    n_iter = max([result[2] for result in all_results])

    if not center_intensity:
        raise ValueError(
            "No point was within bandwidth=%f of any seed. Try a different seeding strategy or increase the bandwidth."
            % fitted_bandwidth
        )

    sorted_by_intensity = sorted(
        center_intensity.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    sorted_centers = np.array([item[0] for item in sorted_by_intensity], dtype=np.float64)
    unique = np.ones(len(sorted_centers), dtype=bool)
    center_neighbors = NearestNeighbors(radius=fitted_bandwidth, n_jobs=n_jobs).fit(sorted_centers)
    for center_index, center in enumerate(sorted_centers):
        if unique[center_index]:
            neighbor_indices = center_neighbors.radius_neighbors([center], return_distance=False)[0]
            unique[neighbor_indices] = 0
            unique[center_index] = 1
    cluster_centers = sorted_centers[unique]

    label_neighbors = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs).fit(cluster_centers)
    labels = np.zeros(n_samples, dtype=np.int_)
    distances, indices = label_neighbors.kneighbors(checked_x)
    if cluster_all:
        labels = indices.flatten().astype(np.int_)
    else:
        labels.fill(-1)
        within_bandwidth = distances.flatten() <= fitted_bandwidth
        labels[within_bandwidth] = indices.flatten()[within_bandwidth]

    return MeanShiftState(
        cluster_centers=np.asarray(cluster_centers, dtype=np.float64),
        labels=labels,
        bandwidth=fitted_bandwidth,
        n_iter=int(n_iter),
        cluster_all=bool(cluster_all),
        n_features_in=int(n_features),
    )


def _kmeans_plusplus_core(
    X: MatrixLike,
    n_clusters: int,
    *,
    sample_weight: NDArray[np.float64] | list[float] | None,
    x_squared_norms: NDArray[np.float64] | list[float] | None,
    random_state: RandomStateLike,
    n_local_trials: int | None,
) -> KMeansPlusPlusResult:
    checked_x = check_array(X, accept_sparse="csr", dtype=np.float64)
    checked_weight = _check_sample_weight(sample_weight, checked_x, dtype=np.float64)
    if x_squared_norms is None:
        squared_norms = row_norms(checked_x, squared=True)
    else:
        squared_norms = check_array(x_squared_norms, dtype=np.float64, ensure_2d=False)
    if squared_norms.shape[0] != checked_x.shape[0]:
        raise ValueError("x_squared_norms length must equal sample count")

    rng = check_random_state(random_state)
    n_samples, n_features = checked_x.shape
    centers = np.empty((n_clusters, n_features), dtype=np.float64)
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    center_id = rng.choice(n_samples, p=checked_weight / checked_weight.sum())
    indices = np.full(n_clusters, -1, dtype=np.int_)
    if sp.issparse(checked_x):
        centers[0] = checked_x[[center_id]].toarray()
    else:
        centers[0] = checked_x[center_id]
    indices[0] = int(center_id)

    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis],
        checked_x,
        Y_norm_squared=squared_norms,
        squared=True,
    )
    current_potential = closest_dist_sq @ checked_weight

    for center_index in range(1, n_clusters):
        rand_vals = rng.uniform(size=n_local_trials) * current_potential
        candidate_ids = np.searchsorted(np.cumsum(checked_weight * closest_dist_sq), rand_vals)
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        distance_to_candidates = euclidean_distances(
            checked_x[candidate_ids],
            checked_x,
            Y_norm_squared=squared_norms,
            squared=True,
        )
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_potential = distance_to_candidates @ checked_weight.reshape(-1, 1)

        best_candidate_index = int(np.argmin(candidates_potential))
        current_potential = candidates_potential[best_candidate_index]
        closest_dist_sq = distance_to_candidates[best_candidate_index]
        best_candidate = int(candidate_ids[best_candidate_index])

        if sp.issparse(checked_x):
            centers[center_index] = checked_x[[best_candidate]].toarray()
        else:
            centers[center_index] = checked_x[best_candidate]
        indices[center_index] = best_candidate

    return centers, indices


def _affinity_propagation_core(
    S: NDArray[np.float64],
    *,
    preference: NDArray[np.float64],
    convergence_iter: int,
    max_iter: int,
    damping: float,
    verbose: bool,
    return_n_iter: bool,
    random_state: np.random.RandomState,
) -> AffinityPropagationOutput:
    n_samples = S.shape[0]
    if n_samples == 1 or _equal_similarities_and_preferences(S, preference):
        warnings.warn(
            "All samples have mutually equal similarities. Returning arbitrary cluster center(s)."
        )
        if preference.flat[0] > S.flat[n_samples - 1]:
            centers = np.arange(n_samples, dtype=np.int_)
            labels = np.arange(n_samples, dtype=np.int_)
        else:
            centers = np.array([0], dtype=np.int_)
            labels = np.array([0] * n_samples, dtype=np.int_)
        if return_n_iter:
            return centers, labels, 0
        return centers, labels

    S.flat[:: n_samples + 1] = preference

    availability = np.zeros((n_samples, n_samples), dtype=np.float64)
    responsibility = np.zeros((n_samples, n_samples), dtype=np.float64)
    tmp = np.zeros((n_samples, n_samples), dtype=np.float64)

    S += (
        np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100
    ) * random_state.standard_normal(size=(n_samples, n_samples))

    exemplars_over_time = np.zeros((n_samples, convergence_iter), dtype=np.float64)
    indices = np.arange(n_samples)
    exemplar_mask = np.zeros(n_samples, dtype=bool)
    never_converged = True
    iteration = 0

    for iteration in range(max_iter):
        np.add(availability, S, tmp)
        best_indices = np.argmax(tmp, axis=1)
        best_values = tmp[indices, best_indices]
        tmp[indices, best_indices] = -np.inf
        second_best_values = np.max(tmp, axis=1)

        np.subtract(S, best_values[:, None], tmp)
        tmp[indices, best_indices] = S[indices, best_indices] - second_best_values

        tmp *= 1.0 - damping
        responsibility *= damping
        responsibility += tmp

        np.maximum(responsibility, 0, out=tmp)
        tmp.flat[:: n_samples + 1] = responsibility.flat[:: n_samples + 1]

        tmp -= np.sum(tmp, axis=0)
        diagonal_availability = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[:: n_samples + 1] = diagonal_availability

        tmp *= 1.0 - damping
        availability *= damping
        availability -= tmp

        exemplar_mask = (np.diag(availability) + np.diag(responsibility)) > 0
        exemplars_over_time[:, iteration % convergence_iter] = exemplar_mask
        n_exemplars = int(np.sum(exemplar_mask, axis=0))

        if iteration >= convergence_iter:
            stability = np.sum(exemplars_over_time, axis=1)
            unconverged = np.sum((stability == convergence_iter) + (stability == 0)) != n_samples
            if not unconverged and n_exemplars > 0:
                never_converged = False
                if verbose:
                    print(f"Converged after {iteration} iterations.")
                break
    else:
        if verbose:
            print("Did not converge")

    exemplar_indices = np.flatnonzero(exemplar_mask)
    n_exemplars = exemplar_indices.size

    if n_exemplars > 0:
        if never_converged:
            warnings.warn(
                "Affinity propagation did not converge, this model may return degenerate cluster centers and labels.",
                ConvergenceWarning,
            )
        cluster_assignments = np.argmax(S[:, exemplar_indices], axis=1)
        cluster_assignments[exemplar_indices] = np.arange(n_exemplars)
        for cluster_index in range(n_exemplars):
            members = np.asarray(cluster_assignments == cluster_index).nonzero()[0]
            best_member = np.argmax(np.sum(S[members[:, np.newaxis], members], axis=0))
            exemplar_indices[cluster_index] = members[best_member]

        cluster_assignments = np.argmax(S[:, exemplar_indices], axis=1)
        cluster_assignments[exemplar_indices] = np.arange(n_exemplars)
        labels_as_indices = exemplar_indices[cluster_assignments]
        cluster_centers_indices = np.unique(labels_as_indices).astype(np.int_)
        labels = np.searchsorted(cluster_centers_indices, labels_as_indices).astype(np.int_)
    else:
        warnings.warn(
            "Affinity propagation did not converge and this model will not have any cluster centers.",
            ConvergenceWarning,
        )
        labels = np.array([-1] * n_samples, dtype=np.int_)
        cluster_centers_indices = np.array([], dtype=np.int_)

    if return_n_iter:
        return cluster_centers_indices, labels, iteration + 1
    return cluster_centers_indices, labels


@register_atom(witness_affinity_propagation)
@icontract.require(lambda S: _is_square_matrix(S), "S must be a square similarity matrix")
@icontract.require(lambda preference: _preference_valid(preference), "preference must be finite scalar or vector")
@icontract.require(lambda S, preference: _preference_matches_samples(preference, S), "preference length must match samples")
@icontract.require(lambda convergence_iter: _positive_int(convergence_iter), "convergence_iter must be at least one")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be at least one")
@icontract.require(lambda damping: _damping_valid(damping), "damping must be in [0.5, 1.0)")
@icontract.ensure(lambda result, S, return_n_iter: _affinity_result_valid(result, S, return_n_iter), "cluster centers and labels must match sample count")
def affinity_propagation(
    S: MatrixLike,
    *,
    preference: PreferenceLike = None,
    convergence_iter: int = 15,
    max_iter: int = 200,
    damping: float = 0.5,
    copy: bool = True,
    verbose: bool = False,
    return_n_iter: bool = False,
    random_state: RandomStateLike = None,
) -> AffinityPropagationOutput:
    """Run affinity propagation on a square similarity matrix."""
    affinity_matrix = np.asarray(
        check_array(S, dtype=[np.float64, np.float32], copy=copy, force_writeable=True),
        dtype=np.float64,
    )
    if affinity_matrix.shape[0] != affinity_matrix.shape[1]:
        raise ValueError("The matrix of similarities must be a square array.")
    prepared_preference = _prepare_preference(preference, affinity_matrix)
    return _affinity_propagation_core(
        affinity_matrix,
        preference=prepared_preference,
        convergence_iter=convergence_iter,
        max_iter=max_iter,
        damping=damping,
        verbose=verbose,
        return_n_iter=return_n_iter,
        random_state=check_random_state(random_state),
    )


@register_atom(witness_affinity_propagation_fit)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a 2D matrix")
@icontract.require(lambda X, affinity: affinity != "precomputed" or _is_square_matrix(X), "precomputed affinity must be square")
@icontract.require(lambda preference: _preference_valid(preference), "preference must be finite scalar or vector")
@icontract.require(lambda X, preference: _preference_matches_samples(preference, X), "preference length must match samples")
@icontract.require(lambda convergence_iter: _positive_int(convergence_iter), "convergence_iter must be at least one")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be at least one")
@icontract.require(lambda damping: _damping_valid(damping), "damping must be in [0.5, 1.0)")
@icontract.require(lambda affinity: _affinity_valid(affinity), "affinity must be 'euclidean' or 'precomputed'")
@icontract.ensure(lambda result: _state_valid(result), "affinity propagation state must be fitted")
def affinity_propagation_fit(
    X: MatrixLike,
    *,
    damping: float = 0.5,
    max_iter: int = 200,
    convergence_iter: int = 15,
    copy: bool = True,
    preference: PreferenceLike = None,
    affinity: str = "euclidean",
    verbose: bool = False,
    random_state: RandomStateLike = None,
) -> AffinityPropagationState:
    """Fit affinity propagation and return immutable clustering state."""
    if affinity == "precomputed":
        affinity_matrix = np.asarray(
            check_array(X, dtype=[np.float64, np.float32], copy=copy, force_writeable=True),
            dtype=np.float64,
        )
        if affinity_matrix.shape[0] != affinity_matrix.shape[1]:
            raise ValueError("The matrix of similarities must be a square array.")
        n_features_in = int(affinity_matrix.shape[1])
        cluster_centers = None
    else:
        checked_x = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        affinity_matrix = np.asarray(-euclidean_distances(checked_x, squared=True), dtype=np.float64)
        n_features_in = int(checked_x.shape[1])
        cluster_centers = np.empty((0, n_features_in), dtype=np.float64)

    prepared_preference = _prepare_preference(preference, affinity_matrix)
    centers, labels, n_iter = _affinity_propagation_core(
        affinity_matrix,
        max_iter=max_iter,
        convergence_iter=convergence_iter,
        preference=prepared_preference,
        damping=damping,
        verbose=verbose,
        return_n_iter=True,
        random_state=check_random_state(random_state),
    )
    centers = np.asarray(centers, dtype=np.int_)
    labels = np.asarray(labels, dtype=np.int_)

    if affinity != "precomputed":
        selected = checked_x[centers].copy()
        cluster_centers = _as_dense_float_matrix(selected)

    return AffinityPropagationState(
        cluster_centers_indices=centers,
        labels=labels,
        n_iter=int(n_iter),
        affinity_matrix=affinity_matrix,
        cluster_centers=cluster_centers,
        affinity=affinity,
        preference=_stored_preference(prepared_preference),
        damping=float(damping),
        n_features_in=n_features_in,
    )


@register_atom(witness_affinity_propagation_predict)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a 2D matrix")
@icontract.require(lambda state: _state_valid(state), "affinity propagation state must be fitted")
@icontract.require(lambda X, state: state.affinity == "precomputed" or _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: _prediction_valid(result, X), "predicted labels must match sample count")
def affinity_propagation_predict(
    X: MatrixLike,
    state: AffinityPropagationState,
) -> NDArray[np.int_]:
    """Assign samples to the nearest fitted affinity-propagation center."""
    if state.affinity == "precomputed":
        raise ValueError("Predict method is not supported when affinity='precomputed'.")
    checked_x = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
    if state.cluster_centers is None:
        raise ValueError("affinity propagation state does not include cluster centers")
    if state.cluster_centers.shape[0] > 0:
        return np.asarray(pairwise_distances_argmin(checked_x, state.cluster_centers), dtype=np.int_)
    warnings.warn(
        "This model does not have any cluster centers because affinity propagation did not converge. Labeling every sample as '-1'.",
        ConvergenceWarning,
    )
    return np.array([-1] * checked_x.shape[0], dtype=np.int_)


@register_atom(witness_estimate_bandwidth)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a 2D matrix")
@icontract.require(lambda quantile: _quantile_valid(quantile), "quantile must be in [0, 1]")
@icontract.require(lambda n_samples: _positive_int_or_none(n_samples), "n_samples must be positive or None")
@icontract.require(lambda n_jobs: _n_jobs_valid(n_jobs), "n_jobs must be an integer or None")
@icontract.ensure(lambda result: _bandwidth_result_valid(result), "estimated bandwidth must be finite and nonnegative")
def estimate_bandwidth(
    X: MatrixLike,
    *,
    quantile: float = 0.3,
    n_samples: int | None = None,
    random_state: RandomStateLike = 0,
    n_jobs: int | None = None,
) -> float:
    """Estimate the flat-kernel bandwidth used by mean-shift clustering."""
    checked_x = np.asarray(check_array(X, dtype=[np.float64, np.float32]), dtype=np.float64)
    rng = check_random_state(random_state)
    if n_samples is not None:
        sample_indices = rng.permutation(checked_x.shape[0])[:n_samples]
        checked_x = checked_x[sample_indices]
    n_neighbors = int(checked_x.shape[0] * quantile)
    if n_neighbors < 1:
        n_neighbors = 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
    nbrs.fit(checked_x)

    bandwidth = 0.0
    for batch in gen_batches(len(checked_x), 500):
        distances, _ = nbrs.kneighbors(checked_x[batch, :], return_distance=True)
        bandwidth += np.max(distances, axis=1).sum()
    return float(bandwidth / checked_x.shape[0])


@register_atom(witness_mean_shift)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a 2D matrix")
@icontract.require(lambda bandwidth: _bandwidth_valid(bandwidth), "bandwidth must be positive or None")
@icontract.require(lambda seeds: _seeds_valid(seeds), "seeds must be a 2D matrix or None")
@icontract.require(lambda X, seeds: _seeds_match_features(seeds, X), "seeds feature count must match X")
@icontract.require(lambda min_bin_freq: _positive_int(min_bin_freq), "min_bin_freq must be at least one")
@icontract.require(lambda max_iter: _nonnegative_int(max_iter), "max_iter must be nonnegative")
@icontract.require(lambda n_jobs: _n_jobs_valid(n_jobs), "n_jobs must be an integer or None")
@icontract.ensure(lambda result, X: _mean_shift_result_valid(result, X), "mean-shift centers and labels must match input shape")
def mean_shift(
    X: MatrixLike,
    *,
    bandwidth: float | None = None,
    seeds: MatrixLike | None = None,
    bin_seeding: bool = False,
    min_bin_freq: int = 1,
    cluster_all: bool = True,
    max_iter: int = 300,
    n_jobs: int | None = None,
) -> MeanShiftResult:
    """Cluster samples by iteratively shifting seed points to local means."""
    state = _mean_shift_fit_core(
        X,
        bandwidth=bandwidth,
        seeds=seeds,
        bin_seeding=bin_seeding,
        min_bin_freq=min_bin_freq,
        cluster_all=cluster_all,
        max_iter=max_iter,
        n_jobs=n_jobs,
    )
    return state.cluster_centers, state.labels


@register_atom(witness_mean_shift_fit)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a 2D matrix")
@icontract.require(lambda bandwidth: _bandwidth_valid(bandwidth), "bandwidth must be positive or None")
@icontract.require(lambda seeds: _seeds_valid(seeds), "seeds must be a 2D matrix or None")
@icontract.require(lambda X, seeds: _seeds_match_features(seeds, X), "seeds feature count must match X")
@icontract.require(lambda min_bin_freq: _positive_int(min_bin_freq), "min_bin_freq must be at least one")
@icontract.require(lambda max_iter: _nonnegative_int(max_iter), "max_iter must be nonnegative")
@icontract.require(lambda n_jobs: _n_jobs_valid(n_jobs), "n_jobs must be an integer or None")
@icontract.ensure(lambda result: _mean_shift_state_valid(result), "mean-shift state must contain fitted centers and labels")
def mean_shift_fit(
    X: MatrixLike,
    *,
    bandwidth: float | None = None,
    seeds: MatrixLike | None = None,
    bin_seeding: bool = False,
    min_bin_freq: int = 1,
    cluster_all: bool = True,
    max_iter: int = 300,
    n_jobs: int | None = None,
) -> MeanShiftState:
    """Fit mean-shift clustering and return immutable cluster state."""
    return _mean_shift_fit_core(
        X,
        bandwidth=bandwidth,
        seeds=seeds,
        bin_seeding=bin_seeding,
        min_bin_freq=min_bin_freq,
        cluster_all=cluster_all,
        max_iter=max_iter,
        n_jobs=n_jobs,
    )


@register_atom(witness_mean_shift_predict)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a 2D matrix")
@icontract.require(lambda state: _mean_shift_state_valid(state), "mean-shift state must contain fitted centers")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: _prediction_valid(result, X), "predicted labels must match sample count")
def mean_shift_predict(
    X: MatrixLike,
    state: MeanShiftState,
) -> NDArray[np.int_]:
    """Assign samples to the nearest fitted mean-shift center."""
    checked_x = np.asarray(check_array(X, dtype=[np.float64, np.float32]), dtype=np.float64)
    return np.asarray(pairwise_distances_argmin(checked_x, state.cluster_centers), dtype=np.int_)


@register_atom(witness_kmeans_plusplus)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a 2D matrix")
@icontract.require(lambda X, n_clusters: _clusters_within_samples(n_clusters, X), "n_clusters must be between one and sample count")
@icontract.require(lambda X, sample_weight: _sample_weight_valid(sample_weight, X), "sample_weight must be nonnegative and match sample count")
@icontract.require(lambda X, x_squared_norms: _vector_matches_samples(x_squared_norms, X), "x_squared_norms must match sample count")
@icontract.require(lambda n_local_trials: _positive_int_or_none(n_local_trials), "n_local_trials must be positive or None")
@icontract.ensure(lambda result, X, n_clusters: _kmeans_plusplus_result_valid(result, X, n_clusters), "k-means++ centers and indices must match requested clusters")
def kmeans_plusplus(
    X: MatrixLike,
    n_clusters: int,
    *,
    sample_weight: NDArray[np.float64] | list[float] | None = None,
    x_squared_norms: NDArray[np.float64] | list[float] | None = None,
    random_state: RandomStateLike = None,
    n_local_trials: int | None = None,
) -> KMeansPlusPlusResult:
    """Choose initial centers with the k-means++ seeding rule."""
    return _kmeans_plusplus_core(
        X,
        n_clusters,
        sample_weight=sample_weight,
        x_squared_norms=x_squared_norms,
        random_state=random_state,
        n_local_trials=n_local_trials,
    )


@register_atom(witness_cluster_optics_dbscan)
@icontract.require(lambda reachability: _is_1d_numeric(reachability), "reachability must be a 1D vector")
@icontract.require(lambda core_distances: _is_1d_numeric(core_distances), "core_distances must be a 1D vector")
@icontract.require(lambda ordering: _is_1d_numeric(ordering), "ordering must be a 1D vector")
@icontract.require(lambda reachability, core_distances, ordering: _same_length_1d(reachability, core_distances, ordering), "OPTICS vectors must have equal length")
@icontract.require(lambda ordering: _ordering_permutation(ordering), "ordering must be a permutation of sample indices")
@icontract.require(lambda eps: _nonnegative_float(eps), "eps must be nonnegative")
@icontract.ensure(lambda result, reachability: _optics_labels_valid(result, reachability), "OPTICS DBSCAN labels must match sample count")
def cluster_optics_dbscan(
    *,
    reachability: NDArray[np.float64],
    core_distances: NDArray[np.float64],
    ordering: NDArray[np.int_],
    eps: float,
) -> NDArray[np.int_]:
    """Extract DBSCAN-style labels from OPTICS reachability arrays."""
    n_samples = len(core_distances)
    labels = np.zeros(n_samples, dtype=np.int_)

    far_reach = reachability > eps
    near_core = core_distances <= eps
    labels[ordering] = np.cumsum(far_reach[ordering] & near_core[ordering]) - 1
    labels[far_reach & ~near_core] = -1
    return labels


def _resolve_optics_size(size: int | float, n_samples: int, name: str) -> int:
    if size > n_samples:
        raise ValueError(f"{name} must be no greater than the number of samples ({n_samples}). Got {size}")
    if size <= 1:
        return max(2, int(size * n_samples))
    return int(size)


def _extend_region(steep_point: NDArray[np.bool_], xward_point: NDArray[np.bool_], start: int, min_samples: int) -> int:
    n_samples = len(steep_point)
    non_xward_points = 0
    index = start
    end = start
    while index < n_samples:
        if steep_point[index]:
            non_xward_points = 0
            end = index
        elif not xward_point[index]:
            non_xward_points += 1
            if non_xward_points > min_samples:
                break
        else:
            return end
        index += 1
    return end


def _update_filter_sdas(
    sdas: list[tuple[int, int, float]],
    mib: float,
    xi_complement: float,
    reachability_plot: NDArray[np.float64],
) -> list[tuple[int, int, float]]:
    if np.isinf(mib):
        return []
    return [
        (start, end, max(stored_mib, mib))
        for start, end, stored_mib in sdas
        if mib <= reachability_plot[start] * xi_complement
    ]


def _correct_predecessor(
    reachability_plot: NDArray[np.float64],
    predecessor_plot: NDArray[np.int_],
    ordering: NDArray[np.int_],
    start: int,
    end: int,
) -> tuple[int | None, int | None]:
    while start < end:
        if reachability_plot[start] > reachability_plot[end]:
            return start, end
        predecessor = predecessor_plot[end]
        for index in range(start, end):
            if predecessor == ordering[index]:
                return start, end
        end -= 1
    return None, None


def _xi_cluster(
    reachability_plot_input: NDArray[np.float64],
    predecessor_plot: NDArray[np.int_],
    ordering: NDArray[np.int_],
    xi: float,
    min_samples: int,
    min_cluster_size: int,
    predecessor_correction: bool,
) -> NDArray[np.int_]:
    reachability_plot = np.hstack((reachability_plot_input, np.inf))
    xi_complement = 1.0 - xi
    sdas: list[tuple[int, int, float]] = []
    clusters: list[tuple[int, int]] = []
    index = 0
    mib = 0.0

    with np.errstate(invalid="ignore"):
        ratio = reachability_plot[:-1] / reachability_plot[1:]
        steep_upward = ratio <= xi_complement
        steep_downward = ratio >= 1.0 / xi_complement if xi_complement != 0.0 else np.full_like(ratio, False, dtype=bool)
        downward = ratio > 1.0
        upward = ratio < 1.0

    for steep_index in iter(np.flatnonzero(steep_upward | steep_downward)):
        if steep_index < index:
            continue

        mib = max(mib, float(np.max(reachability_plot[index : steep_index + 1])))

        if steep_downward[steep_index]:
            sdas = _update_filter_sdas(sdas, mib, xi_complement, reachability_plot)
            down_start = int(steep_index)
            down_end = _extend_region(steep_downward, upward, down_start, min_samples)
            sdas.append((down_start, down_end, 0.0))
            index = down_end + 1
            mib = float(reachability_plot[index])
        else:
            sdas = _update_filter_sdas(sdas, mib, xi_complement, reachability_plot)
            up_start = int(steep_index)
            up_end = _extend_region(steep_upward, downward, up_start, min_samples)
            index = up_end + 1
            mib = float(reachability_plot[index])

            up_clusters: list[tuple[int, int]] = []
            for down_start, down_end, down_mib in sdas:
                cluster_start = down_start
                cluster_end = up_end

                if reachability_plot[cluster_end + 1] * xi_complement < down_mib:
                    continue

                down_max = reachability_plot[down_start]
                if down_max * xi_complement >= reachability_plot[cluster_end + 1]:
                    while reachability_plot[cluster_start + 1] > reachability_plot[cluster_end + 1] and cluster_start < down_end:
                        cluster_start += 1
                elif reachability_plot[cluster_end + 1] * xi_complement >= down_max:
                    while reachability_plot[cluster_end - 1] > down_max and cluster_end > up_start:
                        cluster_end -= 1

                if predecessor_correction:
                    cluster_start_or_none, cluster_end_or_none = _correct_predecessor(
                        reachability_plot,
                        predecessor_plot,
                        ordering,
                        cluster_start,
                        cluster_end,
                    )
                    if cluster_start_or_none is None or cluster_end_or_none is None:
                        continue
                    cluster_start = cluster_start_or_none
                    cluster_end = cluster_end_or_none

                if cluster_end - cluster_start + 1 < min_cluster_size:
                    continue
                if cluster_start > down_end:
                    continue
                if cluster_end < up_start:
                    continue
                up_clusters.append((cluster_start, cluster_end))

            up_clusters.reverse()
            clusters.extend(up_clusters)

    return np.asarray(clusters, dtype=np.int_)


def _extract_xi_labels(ordering: NDArray[np.int_], clusters: NDArray[np.int_]) -> NDArray[np.int_]:
    labels = np.full(len(ordering), -1, dtype=np.int_)
    label = 0
    for cluster in clusters:
        if not np.any(labels[cluster[0] : (cluster[1] + 1)] != -1):
            labels[cluster[0] : (cluster[1] + 1)] = label
            label += 1
    labels[ordering] = labels.copy()
    return labels


@register_atom(witness_cluster_optics_xi)
@icontract.require(lambda reachability: _is_1d_numeric(reachability), "reachability must be a 1D vector")
@icontract.require(lambda predecessor: _is_1d_numeric(predecessor), "predecessor must be a 1D vector")
@icontract.require(lambda ordering: _is_1d_numeric(ordering), "ordering must be a 1D vector")
@icontract.require(lambda reachability, predecessor, ordering: _same_length_1d(reachability, predecessor, ordering), "OPTICS vectors must have equal length")
@icontract.require(lambda ordering: _ordering_permutation(ordering), "ordering must be a permutation of sample indices")
@icontract.require(lambda min_samples, reachability: _optics_size_valid(min_samples, len(reachability), allow_none=False), "min_samples must be valid for sample count")
@icontract.require(lambda min_cluster_size, reachability: _optics_size_valid(min_cluster_size, len(reachability), allow_none=True), "min_cluster_size must be valid for sample count")
@icontract.require(lambda xi: _xi_valid(xi), "xi must be in [0, 1]")
@icontract.ensure(lambda result, reachability: _optics_xi_result_valid(result, reachability), "OPTICS Xi labels and clusters must match sample count")
def cluster_optics_xi(
    *,
    reachability: NDArray[np.float64],
    predecessor: NDArray[np.int_],
    ordering: NDArray[np.int_],
    min_samples: int | float,
    min_cluster_size: int | float | None = None,
    xi: float = 0.05,
    predecessor_correction: bool = True,
) -> OpticsXiResult:
    """Extract clusters from OPTICS reachability using the Xi-steep rule."""
    n_samples = len(reachability)
    resolved_min_samples = _resolve_optics_size(min_samples, n_samples, "min_samples")
    if min_cluster_size is None:
        resolved_min_cluster_size = resolved_min_samples
    else:
        resolved_min_cluster_size = _resolve_optics_size(min_cluster_size, n_samples, "min_cluster_size")

    clusters = _xi_cluster(
        np.asarray(reachability, dtype=np.float64)[ordering],
        np.asarray(predecessor, dtype=np.int_)[ordering],
        np.asarray(ordering, dtype=np.int_),
        float(xi),
        resolved_min_samples,
        resolved_min_cluster_size,
        predecessor_correction,
    )
    if clusters.size == 0:
        clusters = np.empty((0, 2), dtype=np.int_)
    labels = _extract_xi_labels(np.asarray(ordering, dtype=np.int_), clusters)
    return labels, clusters
