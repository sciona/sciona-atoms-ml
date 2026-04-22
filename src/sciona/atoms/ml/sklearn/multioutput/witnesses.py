"""Ghost witnesses for sklearn multioutput helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

ChainOrderSpec = str | tuple[int, ...] | None


def _check_2d(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_multioutput_prediction_matrix(output_predictions: AbstractArray) -> AbstractArray:
    """Describe sklearn's output-by-sample prediction stack as sample-by-output."""
    n_outputs, n_samples = _check_2d(output_predictions, "output_predictions")
    return AbstractArray(shape=(n_samples, n_outputs), dtype="float64")


def witness_multioutput_exact_match_score(
    y_true: AbstractArray,
    y_pred: AbstractArray,
) -> AbstractArray:
    """Describe a scalar exact-match score for multioutput classification."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes")
    _check_2d(y_true, "y_true")
    return AbstractArray(shape=(), dtype="float64", min_val=0.0, max_val=1.0)


def witness_chain_order_indices(
    n_outputs: int,
    *,
    order: ChainOrderSpec = None,
    random_state: int = 0,
) -> AbstractArray:
    """Describe the fitted column order for a classifier or regressor chain."""
    del order, random_state
    if n_outputs < 1:
        raise ValueError("n_outputs must be positive")
    return AbstractArray(shape=(n_outputs,), dtype="int64")


def witness_chain_training_features(
    X: AbstractArray,
    Y: AbstractArray,
    order: AbstractArray,
) -> AbstractArray:
    """Describe training features augmented with targets in chain order."""
    n_samples, n_features = _check_2d(X, "X")
    y_samples, n_outputs = _check_2d(Y, "Y")
    if y_samples != n_samples:
        raise ValueError("X and Y must have matching sample counts")
    if len(order.shape) != 1 or int(order.shape[0]) != n_outputs:
        raise ValueError("order length must match output count")
    return AbstractArray(shape=(n_samples, n_features + n_outputs), dtype="float64")


def witness_chain_step_features(
    X: AbstractArray,
    previous_predictions: AbstractArray,
) -> AbstractArray:
    """Describe features augmented with earlier chain predictions."""
    n_samples, n_features = _check_2d(X, "X")
    prev_samples, n_previous = _check_2d(previous_predictions, "previous_predictions")
    if prev_samples != n_samples:
        raise ValueError("previous predictions must match X sample count")
    return AbstractArray(shape=(n_samples, n_features + n_previous), dtype="float64")


def witness_chain_restore_output_order(
    chain_predictions: AbstractArray,
    order: AbstractArray,
) -> AbstractArray:
    """Describe restoring chain-ordered predictions to original output order."""
    n_samples, n_outputs = _check_2d(chain_predictions, "chain_predictions")
    if len(order.shape) != 1 or int(order.shape[0]) != n_outputs:
        raise ValueError("order length must match output count")
    return AbstractArray(shape=(n_samples, n_outputs), dtype="float64")
