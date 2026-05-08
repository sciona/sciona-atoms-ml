"""Partial-dependence grid helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Iterable

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.stats.mstats import mquantiles

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_feature_axis,
    witness_partial_dependence_grid,
    witness_partial_dependence_grid_parameters,
)

GridResult = tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...]]

def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))

def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))

def _percentiles_valid(percentiles: tuple[float, float]) -> bool:
    return bool(
        isinstance(percentiles, Iterable)
        and len(tuple(percentiles)) == 2
        and all(np.isfinite(float(x)) and 0.0 <= float(x) <= 1.0 for x in percentiles)
        and float(tuple(percentiles)[0]) < float(tuple(percentiles)[1])
    )

def _grid_resolution_valid(grid_resolution: int) -> bool:
    return bool(isinstance(grid_resolution, int) and not isinstance(grid_resolution, bool) and grid_resolution > 1)

def _axis_result_valid(result: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)))

def _grid_result_valid(result: GridResult, X: NDArray[np.float64]) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    grid, axes = result
    values = np.asarray(X, dtype=np.float64)
    grid_values = np.asarray(grid, dtype=np.float64)
    if grid_values.ndim != 2 or grid_values.shape[1] != values.shape[1] or not np.all(np.isfinite(grid_values)):
        return False
    if not isinstance(axes, tuple) or len(axes) != values.shape[1]:
        return False
    axis_lengths: list[int] = []
    for axis in axes:
        axis_values = np.asarray(axis, dtype=np.float64)
        if axis_values.ndim != 1 or axis_values.shape[0] < 1 or not np.all(np.isfinite(axis_values)):
            return False
        axis_lengths.append(axis_values.shape[0])
    return bool(grid_values.shape[0] == int(np.prod(axis_lengths)))

@register_atom(witness_partial_dependence_grid_parameters)
@icontract.require(lambda percentiles: _percentiles_valid(percentiles), "percentiles must be two finite values in [0, 1] with percentiles[0] < percentiles[1]")
@icontract.require(lambda grid_resolution: _grid_resolution_valid(grid_resolution), "grid_resolution must be an integer greater than 1")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2 and _percentiles_valid(result), "validated percentiles must preserve the grid-parameter contract")
def partial_dependence_grid_parameters(
    percentiles: tuple[float, float],
    *,
    grid_resolution: int,
) -> tuple[float, float]:
    """Validate the partial-dependence grid parameters sklearn checks upfront."""
    del grid_resolution
    return float(percentiles[0]), float(percentiles[1])

@register_atom(witness_partial_dependence_feature_axis)
@icontract.require(lambda feature_values: _finite_vector(feature_values), "feature_values must be a nonempty finite vector")
@icontract.require(lambda percentiles: _percentiles_valid(percentiles), "percentiles must be valid")
@icontract.require(lambda grid_resolution: _grid_resolution_valid(grid_resolution), "grid_resolution must be an integer greater than 1")
@icontract.ensure(lambda result: _axis_result_valid(result), "axis must be a finite nonempty vector")
def partial_dependence_feature_axis(
    feature_values: NDArray[np.float64],
    *,
    percentiles: tuple[float, float],
    is_categorical: bool,
    grid_resolution: int,
) -> NDArray[np.float64]:
    """Build one feature axis for partial dependence from unique values or percentiles."""
    validated_percentiles = partial_dependence_grid_parameters(percentiles, grid_resolution=grid_resolution)
    values = np.asarray(feature_values, dtype=np.float64)
    uniques = np.unique(values)
    if bool(is_categorical) or uniques.shape[0] < grid_resolution:
        return np.asarray(uniques, dtype=np.float64)
    empirical = mquantiles(values, prob=validated_percentiles, axis=0)
    empirical = np.asarray(empirical, dtype=np.float64)
    if np.allclose(empirical[0], empirical[1]):
        raise ValueError(
            "percentiles are too close to each other, unable to build the grid. Please choose percentiles that are further apart."
        )
    return np.asarray(
        np.linspace(empirical[0], empirical[1], num=grid_resolution, endpoint=True),
        dtype=np.float64,
    )

@register_atom(witness_partial_dependence_grid)
@icontract.require(lambda X: _finite_matrix(X), "X must be a nonempty finite matrix")
@icontract.require(lambda percentiles: _percentiles_valid(percentiles), "percentiles must be valid")
@icontract.require(lambda grid_resolution: _grid_resolution_valid(grid_resolution), "grid_resolution must be an integer greater than 1")
@icontract.require(
    lambda X, is_categorical: isinstance(is_categorical, tuple) and len(is_categorical) == np.asarray(X).shape[1] and all(isinstance(flag, bool) for flag in is_categorical),
    "is_categorical must be a boolean tuple matching the feature count",
)
@icontract.ensure(lambda result, X: _grid_result_valid(result, X), "grid result must contain a finite Cartesian grid and one finite axis per feature")
def partial_dependence_grid(
    X: NDArray[np.float64],
    *,
    percentiles: tuple[float, float] = (0.05, 0.95),
    is_categorical: tuple[bool, ...],
    grid_resolution: int = 100,
) -> GridResult:
    from sklearn.utils.extmath import cartesian
    """Build the dense partial-dependence grid for supplied numeric feature columns."""
    validated_percentiles = partial_dependence_grid_parameters(percentiles, grid_resolution=grid_resolution)
    values = np.asarray(X, dtype=np.float64)
    axes = tuple(
        partial_dependence_feature_axis(
            values[:, feature],
            percentiles=validated_percentiles,
            is_categorical=is_categorical[feature],
            grid_resolution=grid_resolution,
        )
        for feature in range(values.shape[1])
    )
    return np.asarray(cartesian(axes), dtype=np.float64), axes
