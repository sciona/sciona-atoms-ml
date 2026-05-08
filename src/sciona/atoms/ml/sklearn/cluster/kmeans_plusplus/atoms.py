"""Dense k-means++ seeding atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_kmeans_plusplus_candidate_ids,
    witness_kmeans_plusplus_candidate_potentials,
    witness_kmeans_plusplus_default_local_trials,
    witness_kmeans_plusplus_first_center_index,
    witness_kmeans_plusplus_initialize_dense,
)

CandidatePotentials = tuple[NDArray[np.float64], NDArray[np.float64]]
KMeansPlusPlusInit = tuple[NDArray[np.float64], NDArray[np.int64]]

def _dense_matrix(X: NDArray[np.float64]) -> NDArray[np.float64] | None:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1 or not np.all(np.isfinite(values)):
        return None
    return values

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _random_state_valid(value: int | None) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)

def _weight_vector(values: NDArray[np.float64], n_samples: int) -> NDArray[np.float64] | None:
    try:
        weights = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if weights.ndim != 1 or weights.shape != (n_samples,) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        return None
    if not np.any(weights > 0.0):
        return None
    return weights

def _sample_weight_valid(sample_weight: NDArray[np.float64], n_samples: int) -> bool:
    return _weight_vector(sample_weight, n_samples) is not None

def _distance_vector(values: NDArray[np.float64], n_samples: int) -> NDArray[np.float64] | None:
    try:
        distances = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if distances.ndim != 1 or distances.shape != (n_samples,) or not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
        return None
    return distances

def _closest_dist_sq_valid(closest_dist_sq: NDArray[np.float64], sample_weight: NDArray[np.float64]) -> bool:
    weights = np.asarray(sample_weight, dtype=np.float64)
    distances = _distance_vector(closest_dist_sq, weights.shape[0])
    return distances is not None

def _nonnegative_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0)

def _x_squared_norms_valid(x_squared_norms: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(X, dtype=np.float64)
    distances = _distance_vector(x_squared_norms, values.shape[0])
    return bool(distances is not None)

def _candidate_ids_valid(result: NDArray[np.int64], closest_dist_sq: NDArray[np.float64], n_local_trials: int) -> bool:
    ids = np.asarray(result)
    n_samples = np.asarray(closest_dist_sq, dtype=np.float64).shape[0]
    return bool(
        ids.shape == (n_local_trials,)
        and np.issubdtype(ids.dtype, np.integer)
        and np.all(ids >= 0)
        and np.all(ids < n_samples)
    )

def _candidate_input_valid(
    X: NDArray[np.float64],
    x_squared_norms: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    closest_dist_sq: NDArray[np.float64],
    candidate_ids: NDArray[np.int64],
) -> bool:
    values = _dense_matrix(X)
    if values is None:
        return False
    n_samples = values.shape[0]
    if not (
        _x_squared_norms_valid(x_squared_norms, values)
        and _sample_weight_valid(sample_weight, n_samples)
        and _distance_vector(closest_dist_sq, n_samples) is not None
    ):
        return False
    try:
        ids = np.asarray(candidate_ids, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    return bool(ids.ndim == 1 and ids.shape[0] >= 1 and np.all(ids >= 0) and np.all(ids < n_samples))

def _candidate_potentials_valid(result: CandidatePotentials, X: NDArray[np.float64], candidate_ids: NDArray[np.int64]) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    distance_to_candidates, candidates_pot = result
    values = np.asarray(X, dtype=np.float64)
    ids = np.asarray(candidate_ids, dtype=np.int64)
    distances = np.asarray(distance_to_candidates, dtype=np.float64)
    potentials = np.asarray(candidates_pot, dtype=np.float64)
    return bool(
        distances.shape == (ids.shape[0], values.shape[0])
        and potentials.shape == (ids.shape[0],)
        and np.all(np.isfinite(distances))
        and np.all(np.isfinite(potentials))
        and np.all(distances >= 0.0)
        and np.all(potentials >= 0.0)
    )

def _init_inputs_valid(
    X: NDArray[np.float64],
    n_clusters: int,
    sample_weight: NDArray[np.float64] | None,
    x_squared_norms: NDArray[np.float64] | None,
    random_state: int | None,
    n_local_trials: int | None,
) -> bool:
    values = _dense_matrix(X)
    if values is None or not _positive_int(n_clusters) or not _random_state_valid(random_state):
        return False
    n_samples = values.shape[0]
    if n_clusters > n_samples:
        return False
    if sample_weight is not None and not _sample_weight_valid(sample_weight, n_samples):
        return False
    if x_squared_norms is not None and not _x_squared_norms_valid(x_squared_norms, values):
        return False
    if n_local_trials is not None and not _positive_int(n_local_trials):
        return False
    return True

def _init_result_valid(result: KMeansPlusPlusInit, X: NDArray[np.float64], n_clusters: int) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    centers, indices = result
    values = np.asarray(X, dtype=np.float64)
    center_values = np.asarray(centers, dtype=np.float64)
    index_values = np.asarray(indices)
    return bool(
        center_values.shape == (n_clusters, values.shape[1])
        and index_values.shape == (n_clusters,)
        and np.issubdtype(index_values.dtype, np.integer)
        and np.all(index_values >= 0)
        and np.all(index_values < values.shape[0])
        and np.all(np.isfinite(center_values))
        and np.allclose(center_values, values[index_values])
    )

def _row_squared_norms(X: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(X, dtype=np.float64)
    return np.asarray(np.einsum("ij,ij->i", values, values), dtype=np.float64)

def _stable_cumsum(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.cumsum(np.asarray(values, dtype=np.float64), dtype=np.float64), dtype=np.float64)

def _pairwise_squared_distances_from_candidates(
    X: NDArray[np.float64],
    x_squared_norms: NDArray[np.float64],
    candidate_ids: NDArray[np.int64],
) -> NDArray[np.float64]:
    values = np.asarray(X, dtype=np.float64)
    norms = np.asarray(x_squared_norms, dtype=np.float64)
    ids = np.asarray(candidate_ids, dtype=np.int64)
    candidate_values = values[ids]
    distances = (
        norms[ids][:, np.newaxis]
        + norms[np.newaxis, :]
        - 2.0 * candidate_values @ values.T
    )
    return np.asarray(np.maximum(distances, 0.0), dtype=np.float64)

@register_atom(witness_kmeans_plusplus_default_local_trials)
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "default local trial count must be positive")
def kmeans_plusplus_default_local_trials(n_clusters: int) -> int:
    """Compute sklearn's default greedy-local-trial count for k-means++ seeding."""
    return int(2 + np.log(int(n_clusters)))

@register_atom(witness_kmeans_plusplus_first_center_index)
@icontract.require(lambda sample_weight: _sample_weight_valid(sample_weight, np.asarray(sample_weight, dtype=np.float64).shape[0]), "sample_weight must be a finite nonnegative vector with positive sum")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be None or a nonnegative integer")
@icontract.ensure(lambda result, sample_weight: isinstance(result, int) and 0 <= result < np.asarray(sample_weight, dtype=np.float64).shape[0], "first center index must lie within the sample range")
def kmeans_plusplus_first_center_index(
    sample_weight: NDArray[np.float64],
    *,
    random_state: int | None = None,
) -> int:
    from sklearn.utils import check_random_state
    """Choose the first k-means++ center index by weighted random sampling."""
    weights = np.asarray(sample_weight, dtype=np.float64)
    rng = check_random_state(random_state)
    return int(rng.choice(weights.shape[0], p=weights / weights.sum()))

@register_atom(witness_kmeans_plusplus_candidate_ids)
@icontract.require(lambda closest_dist_sq, sample_weight: _closest_dist_sq_valid(closest_dist_sq, sample_weight), "closest_dist_sq must be a finite nonnegative vector matching sample_weight")
@icontract.require(lambda sample_weight: _sample_weight_valid(sample_weight, np.asarray(sample_weight, dtype=np.float64).shape[0]), "sample_weight must be a finite nonnegative vector with positive sum")
@icontract.require(lambda current_pot: _nonnegative_scalar(current_pot), "current_pot must be finite and nonnegative")
@icontract.require(lambda n_local_trials: _positive_int(n_local_trials), "n_local_trials must be a positive integer")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be None or a nonnegative integer")
@icontract.ensure(lambda result, closest_dist_sq, n_local_trials: _candidate_ids_valid(result, closest_dist_sq, n_local_trials), "candidate ids must be valid sample indices")
def kmeans_plusplus_candidate_ids(
    closest_dist_sq: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    current_pot: float,
    *,
    n_local_trials: int,
    random_state: int | None = None,
) -> NDArray[np.int64]:
    from sklearn.utils import check_random_state
    """Sample candidate center indices for one greedy k-means++ expansion step."""
    distances = np.asarray(closest_dist_sq, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    rng = check_random_state(random_state)
    rand_vals = rng.uniform(size=int(n_local_trials)) * float(current_pot)
    candidate_ids = np.searchsorted(_stable_cumsum(weights * distances), rand_vals)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    np.clip(candidate_ids, None, distances.size - 1, out=candidate_ids)
    return candidate_ids

@register_atom(witness_kmeans_plusplus_candidate_potentials)
@icontract.require(lambda X, x_squared_norms, sample_weight, closest_dist_sq, candidate_ids: _candidate_input_valid(X, x_squared_norms, sample_weight, closest_dist_sq, candidate_ids), "X, norms, weights, distances, and candidate ids must be finite and shape-compatible")
@icontract.ensure(lambda result, X, candidate_ids: _candidate_potentials_valid(result, X, candidate_ids), "candidate potentials must be finite nonnegative arrays with compatible shapes")
def kmeans_plusplus_candidate_potentials(
    X: NDArray[np.float64],
    x_squared_norms: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    closest_dist_sq: NDArray[np.float64],
    candidate_ids: NDArray[np.int64],
) -> CandidatePotentials:
    """Evaluate candidate-center distance updates and resulting weighted potentials."""
    distance_to_candidates = _pairwise_squared_distances_from_candidates(X, x_squared_norms, candidate_ids)
    distance_to_candidates = np.minimum(np.asarray(closest_dist_sq, dtype=np.float64)[np.newaxis, :], distance_to_candidates)
    candidates_pot = np.asarray(distance_to_candidates @ np.asarray(sample_weight, dtype=np.float64).reshape(-1, 1), dtype=np.float64).ravel()
    return np.asarray(distance_to_candidates, dtype=np.float64), candidates_pot

@register_atom(witness_kmeans_plusplus_initialize_dense)
@icontract.require(lambda X, n_clusters, sample_weight, x_squared_norms, random_state, n_local_trials: _init_inputs_valid(X, n_clusters, sample_weight, x_squared_norms, random_state, n_local_trials), "inputs must describe a finite dense dataset with valid k-means++ parameters")
@icontract.ensure(lambda result, X, n_clusters: _init_result_valid(result, X, n_clusters), "initialized centers and indices must be finite, shape-compatible, and sourced from X")
def kmeans_plusplus_initialize_dense(
    X: NDArray[np.float64],
    n_clusters: int,
    *,
    sample_weight: NDArray[np.float64] | None = None,
    x_squared_norms: NDArray[np.float64] | None = None,
    random_state: int | None = None,
    n_local_trials: int | None = None,
) -> KMeansPlusPlusInit:
    from sklearn.utils import check_random_state
    """Initialize dense k-means++ centers and source indices with sklearn's greedy seeding rule."""
    values = np.asarray(X, dtype=np.float64)
    n_samples, n_features = values.shape
    weights = (
        np.asarray(sample_weight, dtype=np.float64)
        if sample_weight is not None
        else np.ones(n_samples, dtype=np.float64)
    )
    norms = (
        np.asarray(x_squared_norms, dtype=np.float64)
        if x_squared_norms is not None
        else _row_squared_norms(values)
    )
    local_trials = int(n_local_trials) if n_local_trials is not None else kmeans_plusplus_default_local_trials(n_clusters)
    rng = check_random_state(random_state)

    centers = np.empty((n_clusters, n_features), dtype=np.float64)
    indices = np.full(n_clusters, -1, dtype=np.int64)

    center_id = int(rng.choice(n_samples, p=weights / weights.sum()))
    centers[0] = values[center_id]
    indices[0] = center_id

    closest_dist_sq = _pairwise_squared_distances_from_candidates(values, norms, np.array([center_id], dtype=np.int64))[0]
    current_pot = float(closest_dist_sq @ weights)

    for c in range(1, int(n_clusters)):
        rand_vals = rng.uniform(size=local_trials) * current_pot
        candidate_ids = np.searchsorted(_stable_cumsum(weights * closest_dist_sq), rand_vals)
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        distance_to_candidates, candidates_pot = kmeans_plusplus_candidate_potentials(
            values,
            norms,
            weights,
            closest_dist_sq,
            candidate_ids,
        )

        best_candidate = int(np.argmin(candidates_pot))
        current_pot = float(candidates_pot[best_candidate])
        closest_dist_sq = np.asarray(distance_to_candidates[best_candidate], dtype=np.float64)
        chosen_index = int(candidate_ids[best_candidate])

        centers[c] = values[chosen_index]
        indices[c] = chosen_index

    return np.asarray(centers, dtype=np.float64), np.asarray(indices, dtype=np.int64)
