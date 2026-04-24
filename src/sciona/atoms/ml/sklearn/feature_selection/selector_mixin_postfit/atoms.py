"""SelectorMixin post-fit helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_selector_feature_names_out,
    witness_selector_inverse_transform_dense,
    witness_selector_support_indices,
    witness_selector_transform_dense,
)


def _support_mask_valid(support_mask: NDArray[np.bool_]) -> bool:
    values = np.asarray(support_mask)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and values.dtype == np.bool_)


def _numeric_matrix_valid(X: NDArray[np.float64]) -> bool:
    values = np.asarray(X, dtype=np.float64)
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _transform_inputs_valid(X: NDArray[np.float64], support_mask: NDArray[np.bool_]) -> bool:
    values = np.asarray(X, dtype=np.float64)
    mask = np.asarray(support_mask)
    return bool(_numeric_matrix_valid(X) and _support_mask_valid(support_mask) and values.shape[1] == mask.shape[0])


def _selected_dense_valid(X_selected: NDArray[np.float64], support_mask: NDArray[np.bool_]) -> bool:
    try:
        values = np.asarray(X_selected, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    mask = np.asarray(support_mask)
    selected_count = int(np.sum(mask)) if _support_mask_valid(support_mask) else -1
    if values.ndim == 1:
        return bool(selected_count == values.shape[0] and np.all(np.isfinite(values)))
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] == selected_count and np.all(np.isfinite(values)))


def _feature_names_valid(input_features: tuple[str, ...], support_mask: NDArray[np.bool_]) -> bool:
    mask = np.asarray(support_mask)
    return bool(
        _support_mask_valid(support_mask)
        and len(input_features) == mask.shape[0]
        and all(isinstance(name, str) and len(name) >= 1 for name in input_features)
    )


def _support_indices_valid(result: NDArray[np.int64], support_mask: NDArray[np.bool_]) -> bool:
    values = np.asarray(result)
    mask = np.asarray(support_mask)
    expected = np.flatnonzero(mask)
    return bool(
        values.ndim == 1
        and np.issubdtype(values.dtype, np.integer)
        and np.array_equal(values, expected)
    )


def _transform_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], support_mask: NDArray[np.bool_]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    X_values = np.asarray(X, dtype=np.float64)
    mask = np.asarray(support_mask)
    return bool(values.shape == (X_values.shape[0], int(np.sum(mask))) and np.all(np.isfinite(values)))


def _inverse_transform_result_valid(result: NDArray[np.float64], X_selected: NDArray[np.float64], support_mask: NDArray[np.bool_]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    mask = np.asarray(support_mask)
    selected = np.asarray(X_selected, dtype=np.float64)
    selected_2d = selected[None, :] if selected.ndim == 1 else selected
    return bool(
        values.shape == (selected_2d.shape[0], mask.shape[0])
        and np.all(np.isfinite(values))
        and np.array_equal(values[:, mask], selected_2d)
        and np.allclose(values[:, ~mask], 0.0)
    )


def _feature_names_result_valid(result: tuple[str, ...], input_features: tuple[str, ...], support_mask: NDArray[np.bool_]) -> bool:
    mask = np.asarray(support_mask)
    expected = tuple(name for name, keep in zip(input_features, mask) if keep)
    return result == expected


@register_atom(witness_selector_support_indices)
@icontract.require(lambda support_mask: _support_mask_valid(support_mask), "support_mask must be a nonempty 1D boolean vector")
@icontract.ensure(lambda result, support_mask: _support_indices_valid(result, support_mask), "support indices must match the true positions in support_mask")
def selector_support_indices(
    support_mask: NDArray[np.bool_],
) -> NDArray[np.int64]:
    """Return selected feature indices from a boolean support mask."""
    return np.asarray(np.flatnonzero(np.asarray(support_mask, dtype=np.bool_)), dtype=np.int64)


@register_atom(witness_selector_transform_dense)
@icontract.require(lambda X, support_mask: _transform_inputs_valid(X, support_mask), "X must be a finite dense matrix matching support_mask length")
@icontract.ensure(lambda result, X, support_mask: _transform_result_valid(result, X, support_mask), "selected matrix must preserve sample count and selected columns")
def selector_transform_dense(
    X: NDArray[np.float64],
    support_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Reduce a dense feature matrix to the columns selected by support_mask."""
    X_values = np.asarray(X, dtype=np.float64)
    mask = np.asarray(support_mask, dtype=np.bool_)
    if not mask.any():
        return np.empty((X_values.shape[0], 0), dtype=X_values.dtype)
    return np.asarray(X_values[:, mask], dtype=np.float64)


@register_atom(witness_selector_inverse_transform_dense)
@icontract.require(lambda X_selected, support_mask: _selected_dense_valid(X_selected, support_mask), "X_selected must be a finite dense matrix or vector matching the selected feature count")
@icontract.require(lambda support_mask: _support_mask_valid(support_mask), "support_mask must be a nonempty 1D boolean vector")
@icontract.ensure(lambda result, X_selected, support_mask: _inverse_transform_result_valid(result, X_selected, support_mask), "inverse transform must restore dropped columns as zeros")
def selector_inverse_transform_dense(
    X_selected: NDArray[np.float64],
    support_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Reinsert zero-filled dropped columns into a dense selected-feature matrix."""
    mask = np.asarray(support_mask, dtype=np.bool_)
    selected = np.asarray(X_selected, dtype=np.float64)
    selected_2d = selected[None, :] if selected.ndim == 1 else selected
    restored = np.zeros((selected_2d.shape[0], mask.shape[0]), dtype=selected_2d.dtype)
    restored[:, mask] = selected_2d
    return np.asarray(restored, dtype=np.float64)


@register_atom(witness_selector_feature_names_out)
@icontract.require(lambda input_features, support_mask: _feature_names_valid(input_features, support_mask), "input_features must be nonempty strings matching support_mask length")
@icontract.ensure(lambda result, input_features, support_mask: _feature_names_result_valid(result, input_features, support_mask), "feature names out must preserve the selected input feature order")
def selector_feature_names_out(
    input_features: tuple[str, ...],
    support_mask: NDArray[np.bool_],
) -> tuple[str, ...]:
    """Mask feature names using a boolean support vector."""
    mask = np.asarray(support_mask, dtype=np.bool_)
    return tuple(name for name, keep in zip(input_features, mask) if keep)
