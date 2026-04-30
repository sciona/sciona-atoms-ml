"""Ghost witnesses for sklearn partial-dependence result packaging."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_grid_value_lengths(grid_values: tuple[AbstractArray, ...]) -> tuple[int, ...]:
    """Describe per-feature grid lengths for partial-dependence outputs."""
    if len(grid_values) < 1:
        raise ValueError("grid_values must be nonempty")
    return tuple(int(values.shape[0]) for values in grid_values)


def witness_partial_dependence_grid_shaped_averages(
    averaged_predictions: AbstractArray,
    grid_value_lengths: tuple[int, ...],
) -> AbstractArray:
    """Describe final average reshaping over grid dimensions."""
    if len(averaged_predictions.shape) != 2:
        raise ValueError("averaged_predictions must be 2D")
    if len(grid_value_lengths) < 1:
        raise ValueError("grid_value_lengths must be nonempty")
    n_points = 1
    for length in grid_value_lengths:
        if length < 1:
            raise ValueError("grid lengths must be positive")
        n_points *= length
    if int(averaged_predictions.shape[1]) != n_points:
        raise ValueError("grid lengths must match the flattened point count")
    return AbstractArray(
        shape=(int(averaged_predictions.shape[0]), *grid_value_lengths),
        dtype="float64",
    )


def witness_partial_dependence_grid_shaped_individual(
    individual_predictions: AbstractArray,
    grid_value_lengths: tuple[int, ...],
) -> AbstractArray:
    """Describe final individual reshaping over grid dimensions."""
    if len(individual_predictions.shape) != 3:
        raise ValueError("individual_predictions must be 3D")
    if len(grid_value_lengths) < 1:
        raise ValueError("grid_value_lengths must be nonempty")
    n_points = 1
    for length in grid_value_lengths:
        if length < 1:
            raise ValueError("grid lengths must be positive")
        n_points *= length
    if int(individual_predictions.shape[2]) != n_points:
        raise ValueError("grid lengths must match the flattened point count")
    return AbstractArray(
        shape=(int(individual_predictions.shape[0]), int(individual_predictions.shape[1]), *grid_value_lengths),
        dtype="float64",
    )


def witness_partial_dependence_result_bunch(
    kind: str,
    grid_values: tuple[AbstractArray, ...],
    average: AbstractArray | None = None,
    individual: AbstractArray | None = None,
) -> dict:
    """Describe the final partial_dependence Bunch payload."""
    if kind not in {"average", "individual", "both"}:
        raise ValueError("kind must be 'average', 'individual', or 'both'")
    if len(grid_values) < 1:
        raise ValueError("grid_values must be nonempty")
    result = {"grid_values": grid_values}
    if kind in {"average", "both"}:
        if average is None:
            raise ValueError("average must be provided when requested")
        result["average"] = average
    if kind in {"individual", "both"}:
        if individual is None:
            raise ValueError("individual must be provided when requested")
        result["individual"] = individual
    return result
