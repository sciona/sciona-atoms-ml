"""Dense MLP optimizer helper atoms adapted from scikit-learn."""

from __future__ import annotations

from typing import Literal

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import AdamOptimizerState, LrScheduleName, SgdOptimizerState, TensorTuple
from .witnesses import (
    witness_mlp_adam_initialize_state,
    witness_mlp_adam_updates,
    witness_mlp_sgd_initialize_state,
    witness_mlp_sgd_iteration_end,
    witness_mlp_sgd_trigger_stopping,
    witness_mlp_sgd_updates,
)

UpdateTuple = tuple[NDArray[np.float64], ...]
SgdUpdateResult = tuple[UpdateTuple, SgdOptimizerState]
AdamUpdateResult = tuple[UpdateTuple, AdamOptimizerState]
StoppingResult = tuple[bool, SgdOptimizerState]

_LR_SCHEDULES = {"constant", "adaptive", "invscaling"}


def _finite_tensor(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim >= 1 and array.size >= 1 and np.all(np.isfinite(array)))


def _tensor_tuple_valid(values: TensorTuple) -> bool:
    return bool(isinstance(values, tuple) and len(values) >= 1 and all(_finite_tensor(value) for value in values))


def _lr_schedule_valid(lr_schedule: str) -> bool:
    return bool(isinstance(lr_schedule, str) and lr_schedule in _LR_SCHEDULES)


def _positive_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)


def _nonnegative_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0)


def _unit_interval_open(value: float) -> bool:
    return bool(_nonnegative_scalar(value) and 0.0 <= float(value) < 1.0)


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _sgd_state_valid(state: SgdOptimizerState) -> bool:
    return bool(
        _positive_scalar(state.learning_rate_init)
        and _positive_scalar(state.learning_rate)
        and _lr_schedule_valid(state.lr_schedule)
        and _nonnegative_scalar(state.momentum)
        and np.isfinite(state.power_t)
        and _tensor_tuple_valid(state.velocities)
    )


def _adam_state_valid(state: AdamOptimizerState) -> bool:
    return bool(
        _positive_scalar(state.learning_rate_init)
        and _positive_scalar(state.learning_rate)
        and _unit_interval_open(state.beta_1)
        and _unit_interval_open(state.beta_2)
        and _positive_scalar(state.epsilon)
        and _nonnegative_int(state.t)
        and _tensor_tuple_valid(state.ms)
        and _tensor_tuple_valid(state.vs)
        and len(state.ms) == len(state.vs)
        and all(np.asarray(m, dtype=np.float64).shape == np.asarray(v, dtype=np.float64).shape for m, v in zip(state.ms, state.vs))
    )


def _grads_align(grads: TensorTuple, tensors: TensorTuple) -> bool:
    return bool(
        _tensor_tuple_valid(grads)
        and len(grads) == len(tensors)
        and all(np.asarray(grad, dtype=np.float64).shape == np.asarray(tensor, dtype=np.float64).shape for grad, tensor in zip(grads, tensors))
    )


def _sgd_init_result_valid(result: SgdOptimizerState, params: TensorTuple) -> bool:
    return bool(
        _sgd_state_valid(result)
        and len(result.velocities) == len(params)
        and all(
            np.asarray(velocity, dtype=np.float64).shape == np.asarray(param, dtype=np.float64).shape
            and np.allclose(np.asarray(velocity, dtype=np.float64), 0.0)
            for velocity, param in zip(result.velocities, params)
        )
    )


def _adam_init_result_valid(result: AdamOptimizerState, params: TensorTuple) -> bool:
    return bool(
        _adam_state_valid(result)
        and len(result.ms) == len(params)
        and len(result.vs) == len(params)
        and all(
            np.asarray(moment, dtype=np.float64).shape == np.asarray(param, dtype=np.float64).shape
            and np.allclose(np.asarray(moment, dtype=np.float64), 0.0)
            for collection in (result.ms, result.vs)
            for moment, param in zip(collection, params)
        )
    )


def _update_tuple_valid(updates: UpdateTuple, reference: TensorTuple) -> bool:
    return bool(
        isinstance(updates, tuple)
        and len(updates) == len(reference)
        and all(
            np.asarray(update, dtype=np.float64).shape == np.asarray(param, dtype=np.float64).shape
            and np.all(np.isfinite(np.asarray(update, dtype=np.float64)))
            for update, param in zip(updates, reference)
        )
    )


def _sgd_update_result_valid(result: SgdUpdateResult, grads: TensorTuple, state: SgdOptimizerState) -> bool:
    updates, next_state = result
    return bool(
        _update_tuple_valid(updates, grads)
        and _sgd_state_valid(next_state)
        and len(next_state.velocities) == len(state.velocities)
        and all(
            np.asarray(velocity, dtype=np.float64).shape == np.asarray(reference, dtype=np.float64).shape
            and np.all(np.isfinite(np.asarray(velocity, dtype=np.float64)))
            for velocity, reference in zip(next_state.velocities, state.velocities)
        )
    )


def _adam_update_result_valid(result: AdamUpdateResult, grads: TensorTuple, state: AdamOptimizerState) -> bool:
    updates, next_state = result
    return bool(
        _update_tuple_valid(updates, grads)
        and _adam_state_valid(next_state)
        and next_state.t == state.t + 1
        and len(next_state.ms) == len(state.ms)
        and len(next_state.vs) == len(state.vs)
    )


def _stopping_result_valid(result: StoppingResult) -> bool:
    should_stop, state = result
    return bool(isinstance(should_stop, bool) and _sgd_state_valid(state))


@register_atom(witness_mlp_sgd_initialize_state)
@icontract.require(lambda params: _tensor_tuple_valid(params), "params must be a nonempty tuple of finite parameter tensors")
@icontract.require(lambda learning_rate_init: _positive_scalar(learning_rate_init), "learning_rate_init must be finite and positive")
@icontract.require(lambda lr_schedule: _lr_schedule_valid(lr_schedule), "lr_schedule must be constant, adaptive, or invscaling")
@icontract.require(lambda momentum: _nonnegative_scalar(momentum), "momentum must be finite and nonnegative")
@icontract.require(lambda power_t: np.isfinite(float(power_t)), "power_t must be finite")
@icontract.ensure(lambda result, params: _sgd_init_result_valid(result, params), "initialized SGD state must have zero velocities aligned with params")
def mlp_sgd_initialize_state(
    params: TensorTuple,
    *,
    learning_rate_init: float = 0.1,
    lr_schedule: LrScheduleName = "constant",
    momentum: float = 0.9,
    nesterov: bool = True,
    power_t: float = 0.5,
) -> SgdOptimizerState:
    """Initialize persistent state for sklearn's SGDOptimizer update kernel."""
    velocities = tuple(np.zeros_like(np.asarray(param, dtype=np.float64)) for param in params)
    return SgdOptimizerState(
        learning_rate_init=float(learning_rate_init),
        learning_rate=float(learning_rate_init),
        lr_schedule=lr_schedule,
        momentum=float(momentum),
        nesterov=bool(nesterov),
        power_t=float(power_t),
        velocities=velocities,
    )


@register_atom(witness_mlp_sgd_updates)
@icontract.require(lambda state: _sgd_state_valid(state), "state must be a valid SGD optimizer state")
@icontract.require(lambda grads, state: _grads_align(grads, state.velocities), "grads must align with SGD velocity tensors")
@icontract.ensure(lambda result, grads, state: _sgd_update_result_valid(result, grads, state), "SGD updates must align with grads and return the next velocity state")
def mlp_sgd_updates(
    grads: TensorTuple,
    state: SgdOptimizerState,
) -> SgdUpdateResult:
    """Compute one SGDOptimizer update step and the next optimizer state."""
    grad_values = tuple(np.asarray(grad, dtype=np.float64) for grad in grads)
    base_updates = tuple(
        state.momentum * np.asarray(velocity, dtype=np.float64) - state.learning_rate * grad
        for velocity, grad in zip(state.velocities, grad_values)
    )
    next_velocities = tuple(np.asarray(update, dtype=np.float64) for update in base_updates)
    if state.nesterov:
        updates = tuple(
            state.momentum * velocity - state.learning_rate * grad
            for velocity, grad in zip(next_velocities, grad_values)
        )
    else:
        updates = next_velocities
    next_state = SgdOptimizerState(
        learning_rate_init=state.learning_rate_init,
        learning_rate=state.learning_rate,
        lr_schedule=state.lr_schedule,
        momentum=state.momentum,
        nesterov=state.nesterov,
        power_t=state.power_t,
        velocities=next_velocities,
    )
    return tuple(np.asarray(update, dtype=np.float64) for update in updates), next_state


@register_atom(witness_mlp_sgd_iteration_end)
@icontract.require(lambda state: _sgd_state_valid(state), "state must be a valid SGD optimizer state")
@icontract.require(lambda time_step: _nonnegative_int(time_step), "time_step must be a nonnegative integer")
@icontract.ensure(lambda result: _sgd_state_valid(result), "iteration-end state must remain a valid SGD optimizer state")
def mlp_sgd_iteration_end(
    state: SgdOptimizerState,
    *,
    time_step: int,
) -> SgdOptimizerState:
    """Apply sklearn's SGDOptimizer iteration-end learning-rate schedule update."""
    if state.lr_schedule != "invscaling":
        return state
    next_learning_rate = float(state.learning_rate_init) / float(time_step + 1) ** float(state.power_t)
    return SgdOptimizerState(
        learning_rate_init=state.learning_rate_init,
        learning_rate=next_learning_rate,
        lr_schedule=state.lr_schedule,
        momentum=state.momentum,
        nesterov=state.nesterov,
        power_t=state.power_t,
        velocities=state.velocities,
    )


@register_atom(witness_mlp_sgd_trigger_stopping)
@icontract.require(lambda state: _sgd_state_valid(state), "state must be a valid SGD optimizer state")
@icontract.ensure(lambda result: _stopping_result_valid(result), "stopping step must return a boolean decision and a valid SGD state")
def mlp_sgd_trigger_stopping(state: SgdOptimizerState) -> StoppingResult:
    """Apply sklearn's SGDOptimizer adaptive-schedule stopping decision without side-effect logging."""
    if state.lr_schedule != "adaptive":
        return True, state
    if state.learning_rate <= 1e-6:
        return True, state
    next_state = SgdOptimizerState(
        learning_rate_init=state.learning_rate_init,
        learning_rate=state.learning_rate / 5.0,
        lr_schedule=state.lr_schedule,
        momentum=state.momentum,
        nesterov=state.nesterov,
        power_t=state.power_t,
        velocities=state.velocities,
    )
    return False, next_state


@register_atom(witness_mlp_adam_initialize_state)
@icontract.require(lambda params: _tensor_tuple_valid(params), "params must be a nonempty tuple of finite parameter tensors")
@icontract.require(lambda learning_rate_init: _positive_scalar(learning_rate_init), "learning_rate_init must be finite and positive")
@icontract.require(lambda beta_1: _unit_interval_open(beta_1), "beta_1 must be in [0, 1)")
@icontract.require(lambda beta_2: _unit_interval_open(beta_2), "beta_2 must be in [0, 1)")
@icontract.require(lambda epsilon: _positive_scalar(epsilon), "epsilon must be finite and positive")
@icontract.ensure(lambda result, params: _adam_init_result_valid(result, params), "initialized Adam state must have zero first and second moments aligned with params")
def mlp_adam_initialize_state(
    params: TensorTuple,
    *,
    learning_rate_init: float = 0.001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
) -> AdamOptimizerState:
    """Initialize persistent state for sklearn's AdamOptimizer update kernel."""
    zeros = tuple(np.zeros_like(np.asarray(param, dtype=np.float64)) for param in params)
    return AdamOptimizerState(
        learning_rate_init=float(learning_rate_init),
        learning_rate=float(learning_rate_init),
        beta_1=float(beta_1),
        beta_2=float(beta_2),
        epsilon=float(epsilon),
        t=0,
        ms=zeros,
        vs=zeros,
    )


@register_atom(witness_mlp_adam_updates)
@icontract.require(lambda state: _adam_state_valid(state), "state must be a valid Adam optimizer state")
@icontract.require(lambda grads, state: _grads_align(grads, state.ms) and _grads_align(grads, state.vs), "grads must align with Adam moment tensors")
@icontract.ensure(lambda result, grads, state: _adam_update_result_valid(result, grads, state), "Adam updates must align with grads and return the next moment state")
def mlp_adam_updates(
    grads: TensorTuple,
    state: AdamOptimizerState,
) -> AdamUpdateResult:
    """Compute one AdamOptimizer update step and the next optimizer state."""
    grad_values = tuple(np.asarray(grad, dtype=np.float64) for grad in grads)
    next_t = state.t + 1
    next_ms = tuple(
        state.beta_1 * np.asarray(m, dtype=np.float64) + (1.0 - state.beta_1) * grad
        for m, grad in zip(state.ms, grad_values)
    )
    next_vs = tuple(
        state.beta_2 * np.asarray(v, dtype=np.float64) + (1.0 - state.beta_2) * (grad**2)
        for v, grad in zip(state.vs, grad_values)
    )
    next_learning_rate = float(
        state.learning_rate_init
        * np.sqrt(1.0 - state.beta_2**next_t)
        / (1.0 - state.beta_1**next_t)
    )
    updates = tuple(
        -next_learning_rate * m / (np.sqrt(v) + state.epsilon)
        for m, v in zip(next_ms, next_vs)
    )
    next_state = AdamOptimizerState(
        learning_rate_init=state.learning_rate_init,
        learning_rate=next_learning_rate,
        beta_1=state.beta_1,
        beta_2=state.beta_2,
        epsilon=state.epsilon,
        t=next_t,
        ms=tuple(np.asarray(m, dtype=np.float64) for m in next_ms),
        vs=tuple(np.asarray(v, dtype=np.float64) for v in next_vs),
    )
    return tuple(np.asarray(update, dtype=np.float64) for update in updates), next_state
