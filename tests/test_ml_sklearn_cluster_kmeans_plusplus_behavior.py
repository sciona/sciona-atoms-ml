from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster._kmeans import _euclidean_distances, _kmeans_plusplus
from sklearn.utils import check_random_state


def _dense_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [4.0, 4.0],
            [5.0, 4.0],
            [4.0, 5.0],
        ],
        dtype=np.float64,
    )
    sample_weight = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 3.0], dtype=np.float64)
    x_squared_norms = np.einsum("ij,ij->i", X, X)
    return X, sample_weight, x_squared_norms


def _stable_cumsum(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.cumsum(np.asarray(values, dtype=np.float64), dtype=np.float64), dtype=np.float64)


def test_kmeans_plusplus_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.kmeans_plusplus import (
        kmeans_plusplus_candidate_ids,
        kmeans_plusplus_candidate_potentials,
        kmeans_plusplus_default_local_trials,
        kmeans_plusplus_first_center_index,
        kmeans_plusplus_initialize_dense,
    )

    assert callable(kmeans_plusplus_candidate_ids)
    assert callable(kmeans_plusplus_candidate_potentials)
    assert callable(kmeans_plusplus_default_local_trials)
    assert callable(kmeans_plusplus_first_center_index)
    assert callable(kmeans_plusplus_initialize_dense)


def test_kmeans_plusplus_default_local_trials_matches_sklearn_formula() -> None:
    from sciona.atoms.ml.sklearn.cluster.kmeans_plusplus import kmeans_plusplus_default_local_trials

    assert kmeans_plusplus_default_local_trials(1) == 2
    assert kmeans_plusplus_default_local_trials(8) == 4


def test_kmeans_plusplus_first_center_index_matches_weighted_choice() -> None:
    from sciona.atoms.ml.sklearn.cluster.kmeans_plusplus import kmeans_plusplus_first_center_index

    _, sample_weight, _ = _dense_dataset()
    seed = 17
    expected = check_random_state(seed).choice(sample_weight.shape[0], p=sample_weight / sample_weight.sum())

    actual = kmeans_plusplus_first_center_index(sample_weight, random_state=seed)

    assert actual == expected


def test_kmeans_plusplus_candidate_ids_match_sampling_formula() -> None:
    from sciona.atoms.ml.sklearn.cluster.kmeans_plusplus import kmeans_plusplus_candidate_ids

    _, sample_weight, _ = _dense_dataset()
    closest_dist_sq = np.array([0.0, 1.0, 1.0, 32.0, 41.0, 41.0], dtype=np.float64)
    current_pot = float(closest_dist_sq @ sample_weight)
    n_local_trials = 4
    seed = 23

    rng = check_random_state(seed)
    rand_vals = rng.uniform(size=n_local_trials) * current_pot
    expected = np.searchsorted(_stable_cumsum(sample_weight * closest_dist_sq), rand_vals)
    expected = np.asarray(expected, dtype=np.int64)
    np.clip(expected, None, closest_dist_sq.size - 1, out=expected)

    actual = kmeans_plusplus_candidate_ids(
        closest_dist_sq,
        sample_weight,
        current_pot,
        n_local_trials=n_local_trials,
        random_state=seed,
    )

    assert np.array_equal(actual, expected)


def test_kmeans_plusplus_candidate_potentials_match_private_distance_update() -> None:
    from sciona.atoms.ml.sklearn.cluster.kmeans_plusplus import kmeans_plusplus_candidate_potentials

    X, sample_weight, x_squared_norms = _dense_dataset()
    center_id = 0
    closest_dist_sq = _euclidean_distances(X[[center_id]], X, Y_norm_squared=x_squared_norms, squared=True)[0]
    candidate_ids = np.array([2, 3, 5], dtype=np.int64)

    actual_distances, actual_potentials = kmeans_plusplus_candidate_potentials(
        X,
        x_squared_norms,
        sample_weight,
        closest_dist_sq,
        candidate_ids,
    )

    expected_distances = _euclidean_distances(
        X[candidate_ids],
        X,
        Y_norm_squared=x_squared_norms,
        squared=True,
    )
    np.minimum(closest_dist_sq, expected_distances, out=expected_distances)
    expected_potentials = (expected_distances @ sample_weight.reshape(-1, 1)).ravel()

    assert np.allclose(actual_distances, expected_distances)
    assert np.allclose(actual_potentials, expected_potentials)


def test_kmeans_plusplus_initialize_dense_matches_private_weighted_initializer() -> None:
    from sciona.atoms.ml.sklearn.cluster.kmeans_plusplus import kmeans_plusplus_initialize_dense

    X, sample_weight, x_squared_norms = _dense_dataset()
    seed = 31
    n_clusters = 3
    n_local_trials = 5

    expected_centers, expected_indices = _kmeans_plusplus(
        X,
        n_clusters,
        x_squared_norms,
        sample_weight,
        check_random_state(seed),
        n_local_trials=n_local_trials,
    )
    actual_centers, actual_indices = kmeans_plusplus_initialize_dense(
        X,
        n_clusters,
        sample_weight=sample_weight,
        x_squared_norms=x_squared_norms,
        random_state=seed,
        n_local_trials=n_local_trials,
    )

    assert np.allclose(actual_centers, expected_centers)
    assert np.array_equal(actual_indices, expected_indices)


def test_kmeans_plusplus_initialize_dense_matches_public_helper_default_trials() -> None:
    from sciona.atoms.ml.sklearn.cluster.kmeans_plusplus import kmeans_plusplus_initialize_dense

    X, _, _ = _dense_dataset()
    seed = 7

    expected_centers, expected_indices = kmeans_plusplus(
        X,
        n_clusters=3,
        random_state=seed,
    )
    actual_centers, actual_indices = kmeans_plusplus_initialize_dense(
        X,
        3,
        random_state=seed,
    )

    assert np.allclose(actual_centers, expected_centers)
    assert np.array_equal(actual_indices, expected_indices)


def test_kmeans_plusplus_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.kmeans_plusplus import (
        kmeans_plusplus_candidate_ids,
        kmeans_plusplus_candidate_potentials,
        kmeans_plusplus_first_center_index,
        kmeans_plusplus_initialize_dense,
    )

    X, sample_weight, x_squared_norms = _dense_dataset()

    with pytest.raises(ViolationError):
        kmeans_plusplus_first_center_index(np.array([0.0, 0.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        kmeans_plusplus_candidate_ids(
            np.array([0.0, -1.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            1.0,
            n_local_trials=2,
            random_state=0,
        )

    with pytest.raises(ViolationError):
        kmeans_plusplus_candidate_potentials(
            X,
            x_squared_norms,
            sample_weight,
            np.ones(X.shape[0] - 1, dtype=np.float64),
            np.array([1, 2], dtype=np.int64),
        )

    with pytest.raises(ViolationError):
        kmeans_plusplus_initialize_dense(X, X.shape[0] + 1)
