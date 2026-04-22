"""Estimator-independent multioutput helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    ChainOrderSpec,
    witness_chain_order_indices,
    witness_chain_restore_output_order,
    witness_chain_step_features,
    witness_chain_training_features,
    witness_multioutput_exact_match_score,
    witness_multioutput_prediction_matrix,
)


def _finite_2d(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _same_shape_2d(left: NDArray[np.float64], right: NDArray[np.float64]) -> bool:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    return bool(_finite_2d(left) and _finite_2d(right) and left_values.shape == right_values.shape)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _order_spec_valid(order: ChainOrderSpec, n_outputs: int) -> bool:
    if order is None:
        return True
    if isinstance(order, str):
        return order == "random"
    if isinstance(order, tuple):
        return len(order) == n_outputs and sorted(order) == list(range(n_outputs))
    return False


def _random_state_valid(random_state: int) -> bool:
    return bool(isinstance(random_state, int) and not isinstance(random_state, bool))


def _order_vector_valid(order: NDArray[np.int64], n_outputs: int) -> bool:
    values = np.asarray(order)
    return bool(
        values.ndim == 1
        and values.shape[0] == n_outputs
        and np.issubdtype(values.dtype, np.integer)
        and sorted(int(value) for value in values) == list(range(n_outputs))
    )


def _training_inputs_valid(X: NDArray[np.float64], Y: NDArray[np.float64], order: NDArray[np.int64]) -> bool:
    x_values = np.asarray(X, dtype=np.float64)
    y_values = np.asarray(Y, dtype=np.float64)
    return bool(
        _finite_2d(X)
        and _finite_2d(Y)
        and x_values.shape[0] == y_values.shape[0]
        and _order_vector_valid(order, int(y_values.shape[1]))
    )


def _step_inputs_valid(X: NDArray[np.float64], previous_predictions: NDArray[np.float64]) -> bool:
    x_values = np.asarray(X, dtype=np.float64)
    previous_values = np.asarray(previous_predictions, dtype=np.float64)
    return bool(_finite_2d(X) and _finite_2d(previous_predictions) and x_values.shape[0] == previous_values.shape[0])


def _prediction_matrix_valid(result: NDArray[np.float64], output_predictions: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(output_predictions, dtype=np.float64)
    return bool(values.shape == (input_values.shape[1], input_values.shape[0]) and np.all(np.isfinite(values)))


def _score_valid(result: float) -> bool:
    return bool(isinstance(result, float) and np.isfinite(result) and 0.0 <= result <= 1.0)


def _training_features_valid(result: NDArray[np.float64], X: NDArray[np.float64], Y: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    x_values = np.asarray(X, dtype=np.float64)
    y_values = np.asarray(Y, dtype=np.float64)
    return bool(values.shape == (x_values.shape[0], x_values.shape[1] + y_values.shape[1]) and np.all(np.isfinite(values)))


def _step_features_valid(result: NDArray[np.float64], X: NDArray[np.float64], previous_predictions: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    x_values = np.asarray(X, dtype=np.float64)
    previous_values = np.asarray(previous_predictions, dtype=np.float64)
    return bool(values.shape == (x_values.shape[0], x_values.shape[1] + previous_values.shape[1]) and np.all(np.isfinite(values)))


def _restored_predictions_valid(result: NDArray[np.float64], chain_predictions: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(chain_predictions, dtype=np.float64)
    return bool(values.shape == input_values.shape and np.all(np.isfinite(values)))


@register_atom(witness_multioutput_prediction_matrix)
@icontract.require(lambda output_predictions: _finite_2d(output_predictions), "output_predictions must be a finite output-by-sample matrix")
@icontract.ensure(lambda result, output_predictions: _prediction_matrix_valid(result, output_predictions), "prediction matrix must be sample by output")
def multioutput_prediction_matrix(output_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transpose per-output predictions into sklearn's sample-by-output matrix."""
    return np.asarray(np.asarray(output_predictions, dtype=np.float64).T, dtype=np.float64)


@register_atom(witness_multioutput_exact_match_score)
@icontract.require(lambda y_true, y_pred: _same_shape_2d(y_true, y_pred), "y_true and y_pred must be finite matrices with matching shapes")
@icontract.ensure(lambda result: _score_valid(result), "score must lie in [0, 1]")
def multioutput_exact_match_score(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Compute sklearn's exact-row-match score for multioutput classification."""
    return float(np.mean(np.all(np.asarray(y_true, dtype=np.float64) == np.asarray(y_pred, dtype=np.float64), axis=1)))


@register_atom(witness_chain_order_indices)
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be positive")
@icontract.require(lambda n_outputs, order: _order_spec_valid(order, n_outputs), "order must be None, 'random', or a full permutation tuple")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be an integer")
@icontract.ensure(lambda result, n_outputs: _order_vector_valid(result, n_outputs), "chain order must be a full output permutation")
def chain_order_indices(
    n_outputs: int,
    *,
    order: ChainOrderSpec = None,
    random_state: int = 0,
) -> NDArray[np.int64]:
    """Resolve sklearn chain order from default, random, or explicit settings."""
    if order is None:
        return np.arange(n_outputs, dtype=np.int64)
    if order == "random":
        return np.asarray(np.random.RandomState(random_state).permutation(n_outputs), dtype=np.int64)
    return np.asarray(order, dtype=np.int64)


@register_atom(witness_chain_training_features)
@icontract.require(lambda X, Y, order: _training_inputs_valid(X, Y, order), "X, Y, and order must have compatible dimensions")
@icontract.ensure(lambda result, X, Y: _training_features_valid(result, X, Y), "training features must append one target column per output")
def chain_training_features(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    order: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Append target columns in fitted chain order for cv=None chain training."""
    x_values = np.asarray(X, dtype=np.float64)
    y_values = np.asarray(Y, dtype=np.float64)
    order_values = np.asarray(order, dtype=np.int64)
    return np.asarray(np.hstack((x_values, y_values[:, order_values])), dtype=np.float64)


@register_atom(witness_chain_step_features)
@icontract.require(lambda X, previous_predictions: _step_inputs_valid(X, previous_predictions), "X and previous_predictions must have matching sample counts")
@icontract.ensure(lambda result, X, previous_predictions: _step_features_valid(result, X, previous_predictions), "step features must append previous prediction columns")
def chain_step_features(
    X: NDArray[np.float64],
    previous_predictions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Append earlier chain predictions to features for a prediction step."""
    return np.asarray(
        np.hstack((np.asarray(X, dtype=np.float64), np.asarray(previous_predictions, dtype=np.float64))),
        dtype=np.float64,
    )


@register_atom(witness_chain_restore_output_order)
@icontract.require(lambda chain_predictions: _finite_2d(chain_predictions), "chain_predictions must be a finite sample-by-output matrix")
@icontract.require(lambda chain_predictions, order: _order_vector_valid(order, np.asarray(chain_predictions).shape[1]), "order must match prediction columns")
@icontract.ensure(lambda result, chain_predictions: _restored_predictions_valid(result, chain_predictions), "restored predictions must keep the same matrix shape")
def chain_restore_output_order(
    chain_predictions: NDArray[np.float64],
    order: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Restore chain-ordered predictions to sklearn's original output-column order."""
    predictions = np.asarray(chain_predictions, dtype=np.float64)
    order_values = np.asarray(order, dtype=np.int64)
    inverse_order = np.empty_like(order_values)
    inverse_order[order_values] = np.arange(len(order_values))
    return np.asarray(predictions[:, inverse_order], dtype=np.float64)
