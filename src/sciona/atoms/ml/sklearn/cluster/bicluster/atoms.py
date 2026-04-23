"""Dense spectral-biclustering preprocessing atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_bistochastic_normalize,
    witness_bicluster_log_normalize,
    witness_bicluster_scale_normalize,
)

BiclusterScaleNormalization = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def _dense_matrix(X: NDArray[np.float64]) -> NDArray[np.float64] | None:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1 or not np.all(np.isfinite(values)):
        return None
    return values


def _make_nonnegative(values: NDArray[np.float64], *, min_value: float = 0.0) -> NDArray[np.float64]:
    minimum = float(values.min())
    if minimum < min_value:
        return values + (min_value - minimum)
    return values.copy()


def _scale_input_valid(X: NDArray[np.float64]) -> bool:
    values = _dense_matrix(X)
    if values is None:
        return False
    shifted = _make_nonnegative(values)
    return bool(np.all(shifted.sum(axis=1) > 0.0) and np.all(shifted.sum(axis=0) > 0.0))


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _positive_float(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)


def _scale_result_valid(result: BiclusterScaleNormalization, X: NDArray[np.float64]) -> bool:
    values = np.asarray(X, dtype=np.float64)
    if not isinstance(result, tuple) or len(result) != 3:
        return False
    normalized, row_diag, col_diag = result
    normalized_values = np.asarray(normalized)
    rows = np.asarray(row_diag)
    cols = np.asarray(col_diag)
    return bool(
        normalized_values.shape == values.shape
        and rows.shape == (values.shape[0],)
        and cols.shape == (values.shape[1],)
        and np.all(np.isfinite(normalized_values))
        and np.all(np.isfinite(rows))
        and np.all(np.isfinite(cols))
        and np.all(normalized_values >= 0.0)
        and np.all(rows > 0.0)
        and np.all(cols > 0.0)
    )


def _matrix_like_result_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(X, dtype=np.float64)
    normalized = np.asarray(result)
    return bool(normalized.shape == values.shape and np.all(np.isfinite(normalized)))


def _nonnegative_matrix_result_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    return bool(_matrix_like_result_valid(result, X) and np.all(np.asarray(result) >= 0.0))


@register_atom(witness_bicluster_scale_normalize)
@icontract.require(lambda X: _scale_input_valid(X), "X must be a finite dense matrix with positive shifted row and column sums")
@icontract.ensure(lambda result, X: _scale_result_valid(result, X), "result must contain a finite normalized matrix and positive row/column scales")
def bicluster_scale_normalize(X: NDArray[np.float64]) -> BiclusterScaleNormalization:
    """Scale rows and columns for spectral co-clustering preprocessing."""
    values = _make_nonnegative(np.asarray(X, dtype=np.float64))
    row_diag = np.asarray(1.0 / np.sqrt(values.sum(axis=1)), dtype=np.float64).squeeze()
    col_diag = np.asarray(1.0 / np.sqrt(values.sum(axis=0)), dtype=np.float64).squeeze()
    normalized = row_diag[:, np.newaxis] * values * col_diag
    return (
        np.asarray(normalized, dtype=np.float64),
        np.asarray(row_diag, dtype=np.float64),
        np.asarray(col_diag, dtype=np.float64),
    )


@register_atom(witness_bicluster_bistochastic_normalize)
@icontract.require(lambda X: _scale_input_valid(X), "X must be a finite dense matrix with positive shifted row and column sums")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be a positive integer")
@icontract.require(lambda tol: _positive_float(tol), "tol must be positive")
@icontract.ensure(lambda result, X: _nonnegative_matrix_result_valid(result, X), "normalized matrix must be finite and nonnegative")
def bicluster_bistochastic_normalize(
    X: NDArray[np.float64],
    *,
    max_iter: int = 1000,
    tol: float = 1e-5,
) -> NDArray[np.float64]:
    """Iteratively normalize spectral biclustering data toward balanced margins."""
    original = _make_nonnegative(np.asarray(X, dtype=np.float64))
    scaled = original
    for _ in range(int(max_iter)):
        scaled_new, _, _ = bicluster_scale_normalize(scaled)
        dist = float(np.linalg.norm(scaled - scaled_new))
        scaled = scaled_new
        if dist < float(tol):
            break
    return np.asarray(scaled, dtype=np.float64)


@register_atom(witness_bicluster_log_normalize)
@icontract.require(lambda X: _dense_matrix(X) is not None, "X must be a finite dense matrix")
@icontract.ensure(lambda result, X: _matrix_like_result_valid(result, X), "log-normalized matrix must be finite with the same shape")
def bicluster_log_normalize(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center log interactions for spectral biclustering preprocessing."""
    values = _make_nonnegative(np.asarray(X, dtype=np.float64), min_value=1.0)
    logs = np.log(values)
    row_avg = logs.mean(axis=1)[:, np.newaxis]
    col_avg = logs.mean(axis=0)
    avg = float(logs.mean())
    return np.asarray(logs - row_avg - col_avg + avg, dtype=np.float64)
