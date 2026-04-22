"""Deterministic IterativeImputer helper atoms adapted from scikit-learn."""

from __future__ import annotations

from typing import Literal

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_iterative_convergence_reached,
    witness_iterative_limit_vector,
    witness_iterative_neighbor_feature_indices,
    witness_iterative_normalized_abs_corr_matrix,
    witness_iterative_ordered_feature_indices,
)

ImputationOrder = Literal["ascending", "descending", "roman", "arabic", "random"]
LimitType = Literal["min", "max"]


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _matrix_with_missing(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array) | np.isnan(array)))


def _bool_matrix(values: NDArray[np.bool_]) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and array.dtype == np.bool_)


def _bool_vector(values: NDArray[np.bool_]) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and array.dtype == np.bool_)


def _finite_positive(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _feature_index_valid(n_features: int, feat_idx: int) -> bool:
    return bool(isinstance(n_features, int) and not isinstance(n_features, bool) and n_features >= 2 and isinstance(feat_idx, int) and not isinstance(feat_idx, bool) and 0 <= feat_idx < n_features)


def _n_nearest_valid(n_nearest_features: int | None, n_features: int) -> bool:
    if n_nearest_features is None:
        return True
    return bool(isinstance(n_nearest_features, int) and not isinstance(n_nearest_features, bool) and 1 <= n_nearest_features <= n_features)


def _abs_corr_valid(abs_corr_mat: NDArray[np.float64] | None, n_features: int, n_nearest_features: int | None) -> bool:
    if n_nearest_features is None or n_nearest_features >= n_features:
        return True
    try:
        values = np.asarray(abs_corr_mat, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.shape == (n_features, n_features)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(values.sum(axis=0), 1.0)
        and np.all(np.diag(values) == 0.0)
    )


def _limit_valid(limit: float | NDArray[np.float64] | None, is_empty_feature: NDArray[np.bool_]) -> bool:
    if limit is None:
        return True
    if np.isscalar(limit):
        return bool(np.isfinite(float(limit)) or np.isinf(float(limit)))
    try:
        values = np.asarray(limit, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] == np.asarray(is_empty_feature).shape[0] and np.all(np.isfinite(values) | np.isinf(values)))


def _same_shape(*arrays: NDArray[np.float64] | NDArray[np.bool_]) -> bool:
    shapes = [np.asarray(array).shape for array in arrays]
    return bool(all(shape == shapes[0] for shape in shapes))


def _ordered_result_valid(result: NDArray[np.int64], mask_missing_values: NDArray[np.bool_], skip_complete: bool) -> bool:
    values = np.asarray(result)
    mask = np.asarray(mask_missing_values)
    n_features = mask.shape[1]
    expected = np.flatnonzero(mask.mean(axis=0)) if skip_complete else np.arange(n_features)
    return bool(values.dtype == np.int64 and set(values.tolist()) == set(expected.tolist()) and values.shape == (expected.shape[0],))


def _corr_result_valid(result: NDArray[np.float64], X_filled: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_features = np.asarray(X_filled).shape[1]
    return bool(
        values.shape == (n_features, n_features)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(np.diag(values) == 0.0)
        and np.allclose(values.sum(axis=0), 1.0)
    )


def _neighbor_result_valid(result: NDArray[np.int64], n_features: int, feat_idx: int, n_nearest_features: int | None) -> bool:
    values = np.asarray(result)
    expected_size = n_features - 1 if n_nearest_features is None or n_nearest_features >= n_features else int(n_nearest_features)
    return bool(
        values.dtype == np.int64
        and values.shape == (expected_size,)
        and np.all((0 <= values) & (values < n_features))
        and feat_idx not in set(values.tolist())
        and len(set(values.tolist())) == values.shape[0]
    )


def _limit_result_valid(result: NDArray[np.float64], is_empty_feature: NDArray[np.bool_], keep_empty_features: bool) -> bool:
    values = np.asarray(result, dtype=np.float64)
    empty = np.asarray(is_empty_feature, dtype=np.bool_)
    expected_size = empty.shape[0] if keep_empty_features else int(np.sum(~empty))
    return bool(values.shape == (expected_size,) and np.all(np.isfinite(values) | np.isinf(values)))


@register_atom(witness_iterative_ordered_feature_indices)
@icontract.require(lambda mask_missing_values: _bool_matrix(mask_missing_values), "mask_missing_values must be a nonempty boolean matrix")
@icontract.require(lambda imputation_order: imputation_order in {"ascending", "descending", "roman", "arabic", "random"}, "imputation_order must be supported")
@icontract.require(lambda random_state: _nonnegative_int(random_state), "random_state must be a nonnegative integer")
@icontract.ensure(lambda result, mask_missing_values, skip_complete: _ordered_result_valid(result, mask_missing_values, skip_complete), "ordered indices must cover the selected features once")
def iterative_ordered_feature_indices(
    mask_missing_values: NDArray[np.bool_],
    *,
    imputation_order: ImputationOrder = "ascending",
    skip_complete: bool = False,
    random_state: int = 0,
) -> NDArray[np.int64]:
    """Choose the feature update order for one IterativeImputer round."""
    mask = np.asarray(mask_missing_values, dtype=np.bool_)
    frac_missing = mask.mean(axis=0)
    if skip_complete:
        missing_idx = np.flatnonzero(frac_missing)
    else:
        missing_idx = np.arange(frac_missing.shape[0])

    if imputation_order == "roman":
        ordered = missing_idx
    elif imputation_order == "arabic":
        ordered = missing_idx[::-1]
    elif imputation_order == "ascending":
        n_complete = len(frac_missing) - len(missing_idx)
        ordered = np.argsort(frac_missing, kind="mergesort")[n_complete:]
    elif imputation_order == "descending":
        n_complete = len(frac_missing) - len(missing_idx)
        ordered = np.argsort(frac_missing, kind="mergesort")[n_complete:][::-1]
    else:
        ordered = missing_idx.copy()
        np.random.RandomState(int(random_state)).shuffle(ordered)
    return np.asarray(ordered, dtype=np.int64)


@register_atom(witness_iterative_normalized_abs_corr_matrix)
@icontract.require(lambda X_filled: _finite_matrix(X_filled) and np.asarray(X_filled).shape[1] >= 2, "X_filled must be finite with at least two features")
@icontract.require(lambda tolerance: _finite_positive(tolerance), "tolerance must be positive")
@icontract.ensure(lambda result, X_filled: _corr_result_valid(result, X_filled), "correlation matrix must be nonnegative and column-normalized")
def iterative_normalized_abs_corr_matrix(
    X_filled: NDArray[np.float64],
    *,
    tolerance: float = 1e-6,
) -> NDArray[np.float64]:
    """Build normalized absolute-correlation probabilities between features."""
    values = np.asarray(X_filled, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        abs_corr = np.abs(np.corrcoef(values.T))
    abs_corr[np.isnan(abs_corr)] = float(tolerance)
    np.clip(abs_corr, float(tolerance), None, out=abs_corr)
    np.fill_diagonal(abs_corr, 0.0)
    column_sums = abs_corr.sum(axis=0)
    return np.asarray(abs_corr / column_sums.reshape((1, -1)), dtype=np.float64)


@register_atom(witness_iterative_neighbor_feature_indices)
@icontract.require(lambda n_features, feat_idx: _feature_index_valid(n_features, feat_idx), "feat_idx must reference one of at least two features")
@icontract.require(lambda n_nearest_features, n_features: _n_nearest_valid(n_nearest_features, n_features), "n_nearest_features must be None or within feature count")
@icontract.require(lambda abs_corr_mat, n_features, n_nearest_features: _abs_corr_valid(abs_corr_mat, n_features, n_nearest_features), "active neighbor sampling requires a normalized correlation matrix")
@icontract.require(lambda random_state: _nonnegative_int(random_state), "random_state must be a nonnegative integer")
@icontract.ensure(lambda result, n_features, feat_idx, n_nearest_features: _neighbor_result_valid(result, n_features, feat_idx, n_nearest_features), "neighbors must be unique valid non-target features")
def iterative_neighbor_feature_indices(
    n_features: int,
    feat_idx: int,
    n_nearest_features: int | None,
    *,
    abs_corr_mat: NDArray[np.float64] | None = None,
    random_state: int = 0,
) -> NDArray[np.int64]:
    """Select neighbor feature indices for one imputed target feature."""
    if n_nearest_features is not None and n_nearest_features < n_features:
        probabilities = np.asarray(abs_corr_mat, dtype=np.float64)[:, int(feat_idx)]
        return np.asarray(
            np.random.RandomState(int(random_state)).choice(
                np.arange(int(n_features)),
                int(n_nearest_features),
                replace=False,
                p=probabilities,
            ),
            dtype=np.int64,
        )
    left = np.arange(int(feat_idx), dtype=np.int64)
    right = np.arange(int(feat_idx) + 1, int(n_features), dtype=np.int64)
    return np.concatenate((left, right)).astype(np.int64)


@register_atom(witness_iterative_limit_vector)
@icontract.require(lambda is_empty_feature: _bool_vector(is_empty_feature), "is_empty_feature must be a boolean vector")
@icontract.require(lambda limit_type: limit_type in {"min", "max"}, "limit_type must be min or max")
@icontract.require(lambda limit, is_empty_feature: _limit_valid(limit, is_empty_feature), "limit must be None, scalar, or one value per original feature")
@icontract.ensure(lambda result, is_empty_feature, keep_empty_features: _limit_result_valid(result, is_empty_feature, keep_empty_features), "limit vector must match kept features")
def iterative_limit_vector(
    limit: float | NDArray[np.float64] | None,
    is_empty_feature: NDArray[np.bool_],
    *,
    limit_type: LimitType,
    keep_empty_features: bool,
) -> NDArray[np.float64]:
    """Expand and filter a min or max bound for iterative imputation."""
    empty = np.asarray(is_empty_feature, dtype=np.bool_)
    default_bound = np.inf if limit_type == "max" else -np.inf
    chosen = default_bound if limit is None else limit
    if np.isscalar(chosen):
        values = np.full(empty.shape[0], float(chosen), dtype=np.float64)
    else:
        values = np.asarray(chosen, dtype=np.float64)
    if not keep_empty_features and values.shape[0] == empty.shape[0]:
        values = values[~empty]
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_iterative_convergence_reached)
@icontract.require(lambda X_current: _finite_matrix(X_current), "X_current must be finite")
@icontract.require(lambda X_previous: _finite_matrix(X_previous), "X_previous must be finite")
@icontract.require(lambda X_original: _matrix_with_missing(X_original), "X_original must be finite apart from missing entries")
@icontract.require(lambda mask_missing_values: _bool_matrix(mask_missing_values), "mask_missing_values must be boolean")
@icontract.require(lambda X_current, X_previous, X_original, mask_missing_values: _same_shape(X_current, X_previous, X_original, mask_missing_values), "all matrices must share shape")
@icontract.require(lambda tol: _finite_positive(tol), "tol must be positive")
@icontract.ensure(lambda result: isinstance(result, bool), "convergence flag must be boolean")
def iterative_convergence_reached(
    X_current: NDArray[np.float64],
    X_previous: NDArray[np.float64],
    X_original: NDArray[np.float64],
    mask_missing_values: NDArray[np.bool_],
    *,
    tol: float,
) -> bool:
    """Apply sklearn's IterativeImputer early-stopping comparison."""
    current = np.asarray(X_current, dtype=np.float64)
    previous = np.asarray(X_previous, dtype=np.float64)
    original = np.asarray(X_original, dtype=np.float64)
    missing_mask = np.asarray(mask_missing_values, dtype=np.bool_)
    observed = np.abs(original[~missing_mask])
    scaled_tolerance = float(tol) * float(np.max(observed))
    change = float(np.linalg.norm(current - previous, ord=np.inf, axis=None))
    return bool(change < scaled_tolerance)
