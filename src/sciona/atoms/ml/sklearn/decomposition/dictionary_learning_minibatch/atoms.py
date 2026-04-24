"""MiniBatch dictionary-learning scheduling and convergence atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dictionary_learning_minibatch_batch_size,
    witness_dictionary_learning_minibatch_component_count,
    witness_dictionary_learning_minibatch_dictionary_change_converged,
    witness_dictionary_learning_minibatch_ewa_cost,
    witness_dictionary_learning_minibatch_fit_algorithm,
    witness_dictionary_learning_minibatch_improvement_state,
    witness_dictionary_learning_minibatch_inner_stats,
    witness_dictionary_learning_minibatch_monitoring_started,
    witness_dictionary_learning_minibatch_stats_decay,
)

StatsPair = tuple[NDArray[np.float64], NDArray[np.float64]]


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _positive_int_or_none(value: int | None) -> bool:
    return value is None or _positive_int(value)


def _finite_scalar(value: float | int) -> bool:
    return bool(not isinstance(value, bool) and np.isscalar(value) and np.isfinite(float(value)))


def _nonnegative_finite_scalar(value: float | int) -> bool:
    return bool(_finite_scalar(value) and float(value) >= 0.0)


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _square_finite_matrix(values: object) -> bool:
    return bool(_finite_matrix(values) and np.asarray(values).shape[0] == np.asarray(values).shape[1])


def _same_shape(X: object, Y: object) -> bool:
    return bool(_finite_matrix(X) and _finite_matrix(Y) and np.asarray(X).shape == np.asarray(Y).shape)


def _inner_stats_inputs_valid(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    X_batch: NDArray[np.float64],
    code: NDArray[np.float64],
    batch_size: int,
    step: int,
) -> bool:
    if not (_square_finite_matrix(A) and _finite_matrix(B) and _finite_matrix(X_batch) and _finite_matrix(code)):
        return False
    a_values = np.asarray(A, dtype=np.float64)
    b_values = np.asarray(B, dtype=np.float64)
    x_values = np.asarray(X_batch, dtype=np.float64)
    code_values = np.asarray(code, dtype=np.float64)
    return bool(
        _positive_int(batch_size)
        and _nonnegative_int(step)
        and x_values.shape[0] == batch_size
        and code_values.shape[0] == batch_size
        and a_values.shape == (code_values.shape[1], code_values.shape[1])
        and b_values.shape == (x_values.shape[1], code_values.shape[1])
    )


def _inner_stats_result_valid(result: StatsPair, A: NDArray[np.float64], B: NDArray[np.float64]) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    next_A, next_B = result
    return bool(
        _square_finite_matrix(next_A)
        and _finite_matrix(next_B)
        and np.asarray(next_A).shape == np.asarray(A).shape
        and np.asarray(next_B).shape == np.asarray(B).shape
        and np.allclose(np.asarray(next_A), np.asarray(next_A).T)
    )


@register_atom(witness_dictionary_learning_minibatch_component_count)
@icontract.require(lambda n_components: n_components is None or _positive_int(n_components), "n_components must be None or a positive integer")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "resolved component count must be positive")
def dictionary_learning_minibatch_component_count(n_components: int | None, n_features: int) -> int:
    """Resolve sklearn's component count for MiniBatchDictionaryLearning."""
    return int(n_features if n_components is None else n_components)


@register_atom(witness_dictionary_learning_minibatch_fit_algorithm)
@icontract.require(lambda fit_algorithm: fit_algorithm in {"lars", "cd"}, "fit_algorithm must be lars or cd")
@icontract.require(lambda positive_code, fit_algorithm: not (positive_code and fit_algorithm == "lars"), "positive_code is not supported for lars coding")
@icontract.ensure(lambda result: result in {"lasso_lars", "lasso_cd"}, "resolved algorithm must be a supported lasso variant")
def dictionary_learning_minibatch_fit_algorithm(fit_algorithm: str, positive_code: bool) -> str:
    """Resolve sklearn's sparse-coding algorithm label for MiniBatchDictionaryLearning."""
    del positive_code
    return "lasso_" + fit_algorithm


@register_atom(witness_dictionary_learning_minibatch_batch_size)
@icontract.require(lambda batch_size: _positive_int(batch_size), "batch_size must be a positive integer")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result, n_samples: _positive_int(result) and result <= n_samples, "resolved batch size must be positive and no larger than n_samples")
def dictionary_learning_minibatch_batch_size(batch_size: int, n_samples: int) -> int:
    """Clamp MiniBatchDictionaryLearning batch size to the available sample count."""
    return int(min(batch_size, n_samples))


@register_atom(witness_dictionary_learning_minibatch_stats_decay)
@icontract.require(lambda batch_size: _positive_int(batch_size), "batch_size must be a positive integer")
@icontract.require(lambda step: _nonnegative_int(step), "step must be a nonnegative integer")
@icontract.ensure(lambda result: _nonnegative_finite_scalar(result) and float(result) <= 1.0, "stats decay must be finite in [0, 1]")
def dictionary_learning_minibatch_stats_decay(batch_size: int, step: int) -> float:
    """Compute sklearn's decay factor for one minibatch sufficient-stat update."""
    if step < batch_size - 1:
        theta = (step + 1) * batch_size
    else:
        theta = batch_size**2 + step + 1 - batch_size
    return float((theta + 1 - batch_size) / (theta + 1))


@register_atom(witness_dictionary_learning_minibatch_inner_stats)
@icontract.require(
    lambda A, B, X_batch, code, batch_size, step: _inner_stats_inputs_valid(A, B, X_batch, code, batch_size, step),
    "A, B, X_batch, code, batch_size, and step must describe a valid minibatch sufficient-stat update",
)
@icontract.ensure(lambda result, A, B: _inner_stats_result_valid(result, A, B), "updated inner statistics must preserve shape and symmetry")
def dictionary_learning_minibatch_inner_stats(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    X_batch: NDArray[np.float64],
    code: NDArray[np.float64],
    *,
    batch_size: int,
    step: int,
) -> StatsPair:
    """Update sklearn's minibatch dictionary-learning sufficient statistics."""
    decay = dictionary_learning_minibatch_stats_decay(batch_size, step)
    next_A = decay * np.asarray(A, dtype=np.float64) + (np.asarray(code, dtype=np.float64).T @ np.asarray(code, dtype=np.float64)) / batch_size
    next_B = decay * np.asarray(B, dtype=np.float64) + (np.asarray(X_batch, dtype=np.float64).T @ np.asarray(code, dtype=np.float64)) / batch_size
    return np.asarray(next_A, dtype=np.float64), np.asarray(next_B, dtype=np.float64)


@register_atom(witness_dictionary_learning_minibatch_monitoring_started)
@icontract.require(lambda step: _nonnegative_int(step), "step must be a nonnegative integer")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda batch_size: _positive_int(batch_size), "batch_size must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, bool), "monitoring flag must be boolean")
def dictionary_learning_minibatch_monitoring_started(step: int, n_samples: int, batch_size: int) -> bool:
    """Check whether sklearn starts convergence monitoring at this zero-based step."""
    one_based_step = step + 1
    return bool(one_based_step > min(100, n_samples / batch_size))


@register_atom(witness_dictionary_learning_minibatch_ewa_cost)
@icontract.require(lambda previous_ewa_cost: previous_ewa_cost is None or _finite_scalar(previous_ewa_cost), "previous_ewa_cost must be None or finite")
@icontract.require(lambda batch_cost: _finite_scalar(batch_cost), "batch_cost must be finite")
@icontract.require(lambda batch_size: _positive_int(batch_size), "batch_size must be a positive integer")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result: _finite_scalar(result), "updated EWA cost must be finite")
def dictionary_learning_minibatch_ewa_cost(
    previous_ewa_cost: float | None,
    batch_cost: float,
    batch_size: int,
    n_samples: int,
) -> float:
    """Update sklearn's exponentially weighted average minibatch cost."""
    if previous_ewa_cost is None:
        return float(batch_cost)
    alpha = min(batch_size / (n_samples + 1), 1.0)
    return float(float(previous_ewa_cost) * (1.0 - alpha) + float(batch_cost) * alpha)


@register_atom(witness_dictionary_learning_minibatch_dictionary_change_converged)
@icontract.require(lambda new_dict: _finite_matrix(new_dict), "new_dict must be a finite 2D matrix")
@icontract.require(lambda new_dict, old_dict: _same_shape(new_dict, old_dict), "old_dict must be a finite 2D matrix with the same shape as new_dict")
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.require(lambda tol: _nonnegative_finite_scalar(tol), "tol must be finite and nonnegative")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def dictionary_learning_minibatch_dictionary_change_converged(
    new_dict: NDArray[np.float64],
    old_dict: NDArray[np.float64],
    *,
    n_components: int,
    tol: float,
) -> bool:
    """Check sklearn's small-dictionary-change stopping rule."""
    dict_diff = float(linalg.norm(np.asarray(new_dict, dtype=np.float64) - np.asarray(old_dict, dtype=np.float64)) / n_components)
    return bool(float(tol) > 0.0 and dict_diff <= float(tol))


@register_atom(witness_dictionary_learning_minibatch_improvement_state)
@icontract.require(lambda ewa_cost: _finite_scalar(ewa_cost), "ewa_cost must be finite")
@icontract.require(lambda ewa_cost_min: ewa_cost_min is None or _finite_scalar(ewa_cost_min), "ewa_cost_min must be None or finite")
@icontract.require(lambda no_improvement: _nonnegative_int(no_improvement), "no_improvement must be a nonnegative integer")
@icontract.require(lambda max_no_improvement: _positive_int_or_none(max_no_improvement) or max_no_improvement == 0, "max_no_improvement must be None or a nonnegative integer")
@icontract.ensure(
    lambda result: isinstance(result, tuple)
    and len(result) == 3
    and _finite_scalar(result[0])
    and _nonnegative_int(result[1])
    and isinstance(result[2], bool),
    "result must be (ewa_cost_min, no_improvement, should_stop) with finite and nonnegative bookkeeping values",
)
def dictionary_learning_minibatch_improvement_state(
    ewa_cost: float,
    ewa_cost_min: float | None,
    no_improvement: int,
    max_no_improvement: int | None,
) -> tuple[float, int, bool]:
    """Update sklearn's minibatch no-improvement counters and stopping flag."""
    current = float(ewa_cost)
    if ewa_cost_min is None or current < float(ewa_cost_min):
        next_min = current
        next_count = 0
    else:
        next_min = float(ewa_cost_min)
        next_count = no_improvement + 1
    should_stop = max_no_improvement is not None and next_count >= max_no_improvement
    return next_min, next_count, should_stop
