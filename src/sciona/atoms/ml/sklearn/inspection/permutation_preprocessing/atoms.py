"""Permutation-importance preprocessing atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from sciona.atoms.ml.sklearn.ensemble.bagging_sampling import bagging_generate_indices
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_permutation_importance_dense_permuted_columns,
    witness_permutation_importance_max_sample_count,
    witness_permutation_importance_row_indices,
    witness_permutation_importance_shuffle_indices,
)

MatrixLike = NDArray[np.float64] | list[list[float]]


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _max_samples_value_valid(max_samples: object, n_samples: int) -> bool:
    if not _positive_int(n_samples):
        return False
    if isinstance(max_samples, int) and not isinstance(max_samples, bool):
        return 1 <= max_samples <= n_samples
    if isinstance(max_samples, float):
        return np.isfinite(max_samples) and 0.0 < max_samples <= 1.0
    return False


def _effective_max_samples_valid(result: object, n_samples: int) -> bool:
    return bool(isinstance(result, int) and 1 <= result <= n_samples)


def _row_indices_valid(result: object, n_population: int, n_samples: int) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (n_samples,)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_population)
        and np.unique(values).shape[0] == n_samples
    )


def _finite_2d_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _shuffle_indices_valid(result: object, n_samples: int, n_repeats: int) -> bool:
    values = np.asarray(result)
    if not (
        values.shape == (n_repeats, n_samples)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_samples)
    ):
        return False
    expected = np.arange(n_samples)
    return all(np.array_equal(np.sort(row), expected) for row in values)


def _shuffle_inputs_valid(X: object, col_idx: int, shuffle_indices: object) -> bool:
    if not _finite_2d_matrix(X):
        return False
    x_values = np.asarray(X, dtype=np.float64)
    indices = np.asarray(shuffle_indices)
    return bool(
        isinstance(col_idx, int)
        and not isinstance(col_idx, bool)
        and 0 <= col_idx < x_values.shape[1]
        and indices.ndim == 2
        and indices.shape[1] == x_values.shape[0]
        and np.issubdtype(indices.dtype, np.integer)
        and np.all(indices >= 0)
        and np.all(indices < x_values.shape[0])
    )


def _permuted_dense_valid(result: object, X: object, shuffle_indices: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    x_values = np.asarray(X, dtype=np.float64)
    indices = np.asarray(shuffle_indices)
    return bool(
        values.shape == (indices.shape[0], x_values.shape[0], x_values.shape[1])
        and np.all(np.isfinite(values))
    )


@register_atom(witness_permutation_importance_max_sample_count)
@icontract.require(lambda max_samples, n_samples: _max_samples_value_valid(max_samples, n_samples), "max_samples must be a positive int within sample count or a float in (0, 1]")
@icontract.ensure(lambda result, n_samples: _effective_max_samples_valid(result, n_samples), "effective max_samples must lie in [1, n_samples]")
def permutation_importance_max_sample_count(max_samples: int | float, n_samples: int) -> int:
    """Resolve sklearn's effective max_samples count."""
    if isinstance(max_samples, int) and not isinstance(max_samples, bool):
        return int(max_samples)
    return int(float(max_samples) * n_samples)


@register_atom(witness_permutation_importance_row_indices)
@icontract.require(lambda n_population: _positive_int(n_population), "n_population must be positive")
@icontract.require(lambda n_population, n_samples: _positive_int(n_samples) and n_samples <= n_population, "n_samples must lie in [1, n_population]")
@icontract.ensure(lambda result, n_population, n_samples: _row_indices_valid(result, n_population, n_samples), "row indices must be an in-range sample-without-replacement draw")
def permutation_importance_row_indices(
    n_population: int,
    n_samples: int,
    *,
    random_state: int = 0,
) -> NDArray[np.int64]:
    """Draw sklearn's permutation-importance row subsample indices."""
    return np.asarray(
        bagging_generate_indices(False, int(n_population), int(n_samples), random_state=int(random_state)),
        dtype=np.int64,
    )


@register_atom(witness_permutation_importance_shuffle_indices)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be positive")
@icontract.require(lambda n_repeats: _positive_int(n_repeats), "n_repeats must be positive")
@icontract.ensure(lambda result, n_samples, n_repeats: _shuffle_indices_valid(result, n_samples, n_repeats), "shuffle indices must contain one row permutation per repeat")
def permutation_importance_shuffle_indices(
    n_samples: int,
    n_repeats: int,
    *,
    random_state: int = 0,
) -> NDArray[np.int64]:
    """Generate sklearn's repeated in-place shuffle index states for one feature."""
    rng = check_random_state(int(random_state))
    shuffling_idx = np.arange(int(n_samples), dtype=np.int64)
    permutations: list[NDArray[np.int64]] = []
    for _ in range(int(n_repeats)):
        rng.shuffle(shuffling_idx)
        permutations.append(np.asarray(shuffling_idx.copy(), dtype=np.int64))
    return np.asarray(permutations, dtype=np.int64)


@register_atom(witness_permutation_importance_dense_permuted_columns)
@icontract.require(lambda X: _finite_2d_matrix(X), "X must be a finite nonempty 2D dense matrix")
@icontract.require(lambda X, col_idx, shuffle_indices: _shuffle_inputs_valid(X, col_idx, shuffle_indices), "col_idx and shuffle_indices must be compatible with X")
@icontract.ensure(lambda result, X, shuffle_indices: _permuted_dense_valid(result, X, shuffle_indices), "permuted dense matrices must preserve repeat, sample, and feature dimensions")
def permutation_importance_dense_permuted_columns(
    X: MatrixLike,
    col_idx: int,
    shuffle_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Apply sklearn's dense column permutation for each supplied shuffle state."""
    X_values = np.asarray(X, dtype=np.float64)
    permutations = np.asarray(shuffle_indices, dtype=np.int64)
    outputs = np.repeat(X_values[np.newaxis, :, :], permutations.shape[0], axis=0)
    original_col = X_values[:, int(col_idx)]
    for repeat_idx, indices in enumerate(permutations):
        outputs[repeat_idx, :, int(col_idx)] = original_col[indices]
    return np.asarray(outputs, dtype=np.float64)
