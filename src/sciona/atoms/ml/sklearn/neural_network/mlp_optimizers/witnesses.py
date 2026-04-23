"""Ghost witnesses for sklearn MLP optimizer helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import AdamOptimizerState, SgdOptimizerState


def witness_mlp_sgd_initialize_state(
    params: tuple[AbstractArray, ...],
    *,
    learning_rate_init: float = 0.1,
    lr_schedule: str = "constant",
    momentum: float = 0.9,
    nesterov: bool = True,
    power_t: float = 0.5,
) -> SgdOptimizerState:
    """Describe SGD state built from given arrays."""
    del learning_rate_init, lr_schedule, momentum, nesterov, power_t
    if not params:
        raise ValueError("params must be nonempty")
    for param in params:
        if len(param.shape) < 1:
            raise ValueError("each param must be at least one-dimensional")
    return SgdOptimizerState(
        learning_rate_init=0.0,
        learning_rate=0.0,
        lr_schedule="constant",
        momentum=0.0,
        nesterov=False,
        power_t=0.0,
        velocities=tuple(AbstractArray(shape=param.shape, dtype="float64") for param in params),
    )


def witness_mlp_sgd_updates(
    grads: tuple[AbstractArray, ...],
    state: SgdOptimizerState,
) -> tuple[tuple[AbstractArray, ...], SgdOptimizerState]:
    """Describe SGD parameter updates and the next optimizer state."""
    if not grads or len(grads) != len(state.velocities):
        raise ValueError("grads must align with state.velocities")
    for grad, velocity in zip(grads, state.velocities):
        if grad.shape != velocity.shape:
            raise ValueError("gradient shapes must match state velocities")
    arrays = tuple(AbstractArray(shape=grad.shape, dtype="float64") for grad in grads)
    next_state = SgdOptimizerState(
        learning_rate_init=state.learning_rate_init,
        learning_rate=state.learning_rate,
        lr_schedule=state.lr_schedule,
        momentum=state.momentum,
        nesterov=state.nesterov,
        power_t=state.power_t,
        velocities=arrays,
    )
    return arrays, next_state


def witness_mlp_sgd_iteration_end(
    state: SgdOptimizerState,
    *,
    time_step: int,
) -> SgdOptimizerState:
    """Describe SGD optimizer state after one iteration-end schedule step."""
    del time_step
    return state


def witness_mlp_sgd_trigger_stopping(
    state: SgdOptimizerState,
) -> tuple[bool, SgdOptimizerState]:
    """Describe the adaptive-schedule stopping decision and next SGD state."""
    return False, state


def witness_mlp_adam_initialize_state(
    params: tuple[AbstractArray, ...],
    *,
    learning_rate_init: float = 0.001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
) -> AdamOptimizerState:
    """Describe Adam optimizer state initialized from supplied parameter tensors."""
    del learning_rate_init, beta_1, beta_2, epsilon
    if not params:
        raise ValueError("params must be nonempty")
    for param in params:
        if len(param.shape) < 1:
            raise ValueError("each param must be at least one-dimensional")
    tensors = tuple(AbstractArray(shape=param.shape, dtype="float64") for param in params)
    return AdamOptimizerState(
        learning_rate_init=0.0,
        learning_rate=0.0,
        beta_1=0.0,
        beta_2=0.0,
        epsilon=0.0,
        t=0,
        ms=tensors,
        vs=tensors,
    )


def witness_mlp_adam_updates(
    grads: tuple[AbstractArray, ...],
    state: AdamOptimizerState,
) -> tuple[tuple[AbstractArray, ...], AdamOptimizerState]:
    """Describe Adam parameter updates and the next optimizer state."""
    if not grads or len(grads) != len(state.ms) or len(grads) != len(state.vs):
        raise ValueError("grads must align with Adam moment tensors")
    for grad, m, v in zip(grads, state.ms, state.vs):
        if grad.shape != m.shape or grad.shape != v.shape:
            raise ValueError("gradient shapes must match Adam moments")
    arrays = tuple(AbstractArray(shape=grad.shape, dtype="float64") for grad in grads)
    next_state = AdamOptimizerState(
        learning_rate_init=state.learning_rate_init,
        learning_rate=state.learning_rate,
        beta_1=state.beta_1,
        beta_2=state.beta_2,
        epsilon=state.epsilon,
        t=state.t,
        ms=arrays,
        vs=arrays,
    )
    return arrays, next_state
