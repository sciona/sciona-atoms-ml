"""Ghost witnesses for partial-dependence brute postprocessing helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def witness_partial_dependence_assign_grid_values(
    X: AbstractArray,
    new_values: AbstractArray,
    *,
    features: tuple[int, ...],
) -> AbstractArray:
    """Describe assigning one grid point into the selected dense feature columns."""
    rows, cols = _check_matrix(X, "X")
    if _check_vector(new_values, "new_values") != len(features):
        raise ValueError("new_values must match features")
    for feature in features:
        if feature < 0 or feature >= cols:
            raise ValueError("features must reference columns in X")
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_partial_dependence_average_response_sequence(
    responses: tuple[AbstractArray, ...],
    *,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe stacked averages over a response sequence."""
    if len(responses) < 1:
        raise ValueError("responses must be nonempty")
    first = responses[0]
    if len(first.shape) == 1:
        n_samples = _check_vector(first, "responses[0]")
        if sample_weight is not None and _check_vector(sample_weight, "sample_weight") != n_samples:
            raise ValueError("sample_weight must match the sample axis")
        return AbstractArray(shape=(len(responses),), dtype="float64")
    n_samples, n_targets = _check_matrix(first, "responses[0]")
    if sample_weight is not None and _check_vector(sample_weight, "sample_weight") != n_samples:
        raise ValueError("sample_weight must match the sample axis")
    return AbstractArray(shape=(n_targets, len(responses)), dtype="float64")


def witness_partial_dependence_stack_response_sequence(
    responses: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe sklearn's transposed response stack for brute partial dependence."""
    if len(responses) < 1:
        raise ValueError("responses must be nonempty")
    first = responses[0]
    if len(first.shape) == 1:
        n_samples = _check_vector(first, "responses[0]")
        return AbstractArray(shape=(n_samples, len(responses)), dtype="float64")
    n_samples, n_targets = _check_matrix(first, "responses[0]")
    return AbstractArray(shape=(n_targets, n_samples, len(responses)), dtype="float64")


def witness_partial_dependence_finalize_predictions(
    stacked_predictions: AbstractArray,
    *,
    task_kind: str,
    n_samples: int,
) -> AbstractArray:
    """Describe the final brute prediction tensor after sklearn's reshape rules."""
    del task_kind
    if len(stacked_predictions.shape) == 2:
        rows, cols = _check_matrix(stacked_predictions, "stacked_predictions")
        if rows != n_samples:
            raise ValueError("2D stacked_predictions must align with n_samples")
        return AbstractArray(shape=(rows, cols), dtype="float64")
    targets = int(stacked_predictions.shape[0])
    samples = int(stacked_predictions.shape[1])
    points = int(stacked_predictions.shape[2])
    if samples != n_samples:
        raise ValueError("3D stacked_predictions must align with n_samples")
    if targets == 2:
        return AbstractArray(shape=(samples, points), dtype="float64")
    return AbstractArray(shape=(targets, samples, points), dtype="float64")


def witness_partial_dependence_finalize_averages(
    stacked_averages: AbstractArray,
    *,
    task_kind: str,
) -> AbstractArray:
    """Describe the final brute average-response array after sklearn's reshape rules."""
    del task_kind
    if len(stacked_averages.shape) == 1:
        points = _check_vector(stacked_averages, "stacked_averages")
        return AbstractArray(shape=(1, points), dtype="float64")
    rows, cols = _check_matrix(stacked_averages, "stacked_averages")
    if rows == 2:
        return AbstractArray(shape=(1, cols), dtype="float64")
    return AbstractArray(shape=(rows, cols), dtype="float64")
