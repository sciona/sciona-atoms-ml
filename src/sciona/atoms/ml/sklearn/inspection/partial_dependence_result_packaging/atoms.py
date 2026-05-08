"""Partial-dependence result-packaging atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_grid_shaped_averages,
    witness_partial_dependence_grid_shaped_individual,
    witness_partial_dependence_grid_value_lengths,
    witness_partial_dependence_result_bunch,
)

Kind = Literal["average", "individual", "both"]
GridValues = tuple[NDArray[np.float64], ...]

def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))

def _grid_values_valid(grid_values: object) -> bool:
    if not isinstance(grid_values, tuple) or len(grid_values) < 1:
        return False
    return all(_finite_vector(values) for values in grid_values)

def _grid_lengths_valid(result: object, grid_values: GridValues) -> bool:
    return bool(
        isinstance(result, tuple)
        and len(result) == len(grid_values)
        and all(isinstance(length, int) and length >= 1 for length in result)
        and result == tuple(int(np.asarray(values).shape[0]) for values in grid_values)
    )

def _positive_grid_lengths(grid_value_lengths: object) -> bool:
    return bool(
        isinstance(grid_value_lengths, tuple)
        and len(grid_value_lengths) >= 1
        and all(isinstance(length, int) and not isinstance(length, bool) and length >= 1 for length in grid_value_lengths)
    )

def _finite_2d(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))

def _finite_3d(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 3
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and array.shape[2] >= 1
        and np.all(np.isfinite(array))
    )

def _flattened_point_count(grid_value_lengths: tuple[int, ...]) -> int:
    total = 1
    for length in grid_value_lengths:
        total *= int(length)
    return total

def _grid_shaped_averages_valid(result: object, averaged_predictions: object, grid_value_lengths: tuple[int, ...]) -> bool:
    if not (_finite_2d(averaged_predictions) and _positive_grid_lengths(grid_value_lengths)):
        return False
    values = np.asarray(result, dtype=np.float64)
    averages = np.asarray(averaged_predictions, dtype=np.float64)
    return bool(
        averages.shape[1] == _flattened_point_count(grid_value_lengths)
        and values.shape == (averages.shape[0], *grid_value_lengths)
        and np.all(np.isfinite(values))
    )

def _grid_shaped_individual_valid(result: object, individual_predictions: object, grid_value_lengths: tuple[int, ...]) -> bool:
    if not (_finite_3d(individual_predictions) and _positive_grid_lengths(grid_value_lengths)):
        return False
    values = np.asarray(result, dtype=np.float64)
    individual = np.asarray(individual_predictions, dtype=np.float64)
    return bool(
        individual.shape[2] == _flattened_point_count(grid_value_lengths)
        and values.shape == (individual.shape[0], individual.shape[1], *grid_value_lengths)
        and np.all(np.isfinite(values))
    )

def _kind_valid(kind: object) -> bool:
    return kind in {"average", "individual", "both"}

def _result_bunch_valid(
    result: object,
    kind: Kind,
    grid_values: GridValues,
    average: object | None,
    individual: object | None,
) -> bool:
    from sklearn.utils import Bunch
    if not isinstance(result, Bunch):
        return False
    if result.get("grid_values", None) != grid_values:
        return False
    if kind == "average":
        return "average" in result and "individual" not in result and np.array_equal(np.asarray(result["average"]), np.asarray(average))
    if kind == "individual":
        return "individual" in result and "average" not in result and np.array_equal(np.asarray(result["individual"]), np.asarray(individual))
    return (
        "average" in result
        and "individual" in result
        and np.array_equal(np.asarray(result["average"]), np.asarray(average))
        and np.array_equal(np.asarray(result["individual"]), np.asarray(individual))
    )

@register_atom(witness_partial_dependence_grid_value_lengths)
@icontract.require(lambda grid_values: _grid_values_valid(grid_values), "grid_values must be a nonempty tuple of finite nonempty vectors")
@icontract.ensure(lambda result, grid_values: _grid_lengths_valid(result, grid_values), "grid lengths must match the grid value vectors")
def partial_dependence_grid_value_lengths(grid_values: GridValues) -> tuple[int, ...]:
    """Return sklearn's per-feature grid lengths from the grid_values tuple."""
    return tuple(int(np.asarray(values, dtype=np.float64).shape[0]) for values in grid_values)

@register_atom(witness_partial_dependence_grid_shaped_averages)
@icontract.require(lambda averaged_predictions: _finite_2d(averaged_predictions), "averaged_predictions must be a finite 2D array")
@icontract.require(lambda grid_value_lengths: _positive_grid_lengths(grid_value_lengths), "grid_value_lengths must be a nonempty tuple of positive integers")
@icontract.require(lambda averaged_predictions, grid_value_lengths: np.asarray(averaged_predictions).shape[1] == _flattened_point_count(grid_value_lengths), "grid lengths must match the flattened average point count")
@icontract.ensure(lambda result, averaged_predictions, grid_value_lengths: _grid_shaped_averages_valid(result, averaged_predictions, grid_value_lengths), "reshaped averages must match sklearn's grid-shaped output")
def partial_dependence_grid_shaped_averages(
    averaged_predictions: NDArray[np.float64],
    grid_value_lengths: tuple[int, ...],
) -> NDArray[np.float64]:
    """Reshape finalized average predictions to sklearn's grid-shaped output."""
    averages = np.asarray(averaged_predictions, dtype=np.float64)
    return np.asarray(averages.reshape(-1, *grid_value_lengths), dtype=np.float64)

@register_atom(witness_partial_dependence_grid_shaped_individual)
@icontract.require(lambda individual_predictions: _finite_3d(individual_predictions), "individual_predictions must be a finite 3D array")
@icontract.require(lambda grid_value_lengths: _positive_grid_lengths(grid_value_lengths), "grid_value_lengths must be a nonempty tuple of positive integers")
@icontract.require(lambda individual_predictions, grid_value_lengths: np.asarray(individual_predictions).shape[2] == _flattened_point_count(grid_value_lengths), "grid lengths must match the flattened individual point count")
@icontract.ensure(lambda result, individual_predictions, grid_value_lengths: _grid_shaped_individual_valid(result, individual_predictions, grid_value_lengths), "reshaped individual predictions must match sklearn's grid-shaped output")
def partial_dependence_grid_shaped_individual(
    individual_predictions: NDArray[np.float64],
    grid_value_lengths: tuple[int, ...],
) -> NDArray[np.float64]:
    """Reshape finalized individual predictions to sklearn's grid-shaped output."""
    individual = np.asarray(individual_predictions, dtype=np.float64)
    return np.asarray(individual.reshape(individual.shape[0], individual.shape[1], *grid_value_lengths), dtype=np.float64)

@register_atom(witness_partial_dependence_result_bunch)
@icontract.require(lambda kind: _kind_valid(kind), "kind must be 'average', 'individual', or 'both'")
@icontract.require(lambda grid_values: _grid_values_valid(grid_values), "grid_values must be a nonempty tuple of finite nonempty vectors")
@icontract.require(lambda kind, average=None, individual=None: (kind not in {"average", "both"} or average is not None) and (kind not in {"individual", "both"} or individual is not None), "requested outputs must be provided")
@icontract.ensure(lambda result, kind, grid_values, average=None, individual=None: _result_bunch_valid(result, kind, grid_values, average, individual), "result Bunch must contain the expected sklearn partial_dependence keys")
def partial_dependence_result_bunch(
    kind: Kind,
    grid_values: GridValues,
    *,
    average: NDArray[np.float64] | None = None,
    individual: NDArray[np.float64] | None = None,
) -> Bunch:
    from sklearn.utils import Bunch
    """Build sklearn's final partial_dependence Bunch from packaged outputs."""
    result = Bunch(grid_values=grid_values)
    if kind in {"average", "both"}:
        result["average"] = np.asarray(average, dtype=np.float64)
    if kind in {"individual", "both"}:
        result["individual"] = np.asarray(individual, dtype=np.float64)
    return result
