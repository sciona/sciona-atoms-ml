from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.utils import check_random_state
from sklearn.ensemble._bagging import _generate_indices


def _data() -> np.ndarray:
    return np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
            [5.0, 50.0, 500.0],
        ],
        dtype=np.float64,
    )


def test_permutation_preprocessing_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_preprocessing import (
        permutation_importance_dense_permuted_columns,
        permutation_importance_max_sample_count,
        permutation_importance_row_indices,
        permutation_importance_shuffle_indices,
    )

    assert callable(permutation_importance_max_sample_count)
    assert callable(permutation_importance_row_indices)
    assert callable(permutation_importance_shuffle_indices)
    assert callable(permutation_importance_dense_permuted_columns)


def test_permutation_preprocessing_matches_sklearn_sampling_and_shuffling() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_preprocessing import (
        permutation_importance_dense_permuted_columns,
        permutation_importance_max_sample_count,
        permutation_importance_row_indices,
        permutation_importance_shuffle_indices,
    )

    X = _data()
    effective = permutation_importance_max_sample_count(0.6, X.shape[0])
    row_indices = permutation_importance_row_indices(X.shape[0], effective, random_state=7)
    expected_row_indices = _generate_indices(
        random_state=check_random_state(7),
        bootstrap=False,
        n_population=X.shape[0],
        n_samples=effective,
    )

    shuffles = permutation_importance_shuffle_indices(X.shape[0], 3, random_state=11)
    rng = check_random_state(11)
    manual_idx = np.arange(X.shape[0])
    expected_shuffles = []
    for _ in range(3):
        rng.shuffle(manual_idx)
        expected_shuffles.append(manual_idx.copy())
    expected_shuffles = np.asarray(expected_shuffles, dtype=np.int64)

    permuted = permutation_importance_dense_permuted_columns(X, 1, shuffles)
    expected_permuted = np.repeat(X[np.newaxis, :, :], 3, axis=0)
    original_col = X[:, 1].copy()
    for repeat_idx, indices in enumerate(expected_shuffles):
        expected_permuted[repeat_idx, :, 1] = original_col[indices]

    assert effective == 3
    assert np.array_equal(row_indices, expected_row_indices.astype(np.int64))
    assert np.array_equal(shuffles, expected_shuffles)
    assert np.array_equal(permuted, expected_permuted)


def test_permutation_preprocessing_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_preprocessing import (
        permutation_importance_dense_permuted_columns,
        permutation_importance_max_sample_count,
        permutation_importance_row_indices,
        permutation_importance_shuffle_indices,
    )

    with pytest.raises(ViolationError):
        permutation_importance_max_sample_count(2.0, 5)

    with pytest.raises(ViolationError):
        permutation_importance_row_indices(5, 6, random_state=0)

    with pytest.raises(ViolationError):
        permutation_importance_shuffle_indices(0, 3, random_state=0)

    with pytest.raises(ViolationError):
        permutation_importance_dense_permuted_columns(_data(), 5, np.arange(5, dtype=np.int64).reshape(1, 5))
