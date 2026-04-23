from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer


def _params() -> tuple[np.ndarray, ...]:
    return (
        np.array([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]], dtype=np.float64),
        np.array([0.1, -0.2, 0.3], dtype=np.float64),
    )


def _grads() -> tuple[np.ndarray, ...]:
    return (
        np.array([[0.25, -0.5, 0.1], [0.4, -0.2, 0.3]], dtype=np.float64),
        np.array([-0.1, 0.2, 0.05], dtype=np.float64),
    )


def test_mlp_optimizer_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_optimizers import (
        mlp_adam_initialize_state,
        mlp_adam_updates,
        mlp_sgd_initialize_state,
        mlp_sgd_iteration_end,
        mlp_sgd_trigger_stopping,
        mlp_sgd_updates,
    )

    assert callable(mlp_sgd_initialize_state)
    assert callable(mlp_sgd_updates)
    assert callable(mlp_sgd_iteration_end)
    assert callable(mlp_sgd_trigger_stopping)
    assert callable(mlp_adam_initialize_state)
    assert callable(mlp_adam_updates)


def test_mlp_sgd_initialize_state_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_optimizers import mlp_sgd_initialize_state

    params = _params()
    state = mlp_sgd_initialize_state(
        params,
        learning_rate_init=0.2,
        lr_schedule="adaptive",
        momentum=0.8,
        nesterov=False,
        power_t=0.3,
    )
    expected = SGDOptimizer(
        list(params),
        learning_rate_init=0.2,
        lr_schedule="adaptive",
        momentum=0.8,
        nesterov=False,
        power_t=0.3,
    )

    assert state.learning_rate_init == expected.learning_rate_init
    assert state.learning_rate == expected.learning_rate
    assert state.lr_schedule == expected.lr_schedule
    assert state.momentum == expected.momentum
    assert state.nesterov == expected.nesterov
    assert state.power_t == expected.power_t
    for actual, target in zip(state.velocities, expected.velocities):
        assert np.array_equal(actual, target)


def test_mlp_sgd_updates_match_sklearn_without_nesterov() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_optimizers import mlp_sgd_initialize_state, mlp_sgd_updates

    params = _params()
    grads = _grads()
    state = mlp_sgd_initialize_state(params, learning_rate_init=0.2, lr_schedule="constant", momentum=0.9, nesterov=False)
    updates, next_state = mlp_sgd_updates(grads, state)

    expected = SGDOptimizer(list(params), learning_rate_init=0.2, lr_schedule="constant", momentum=0.9, nesterov=False)
    target_updates = expected._get_updates(list(grads))

    for actual, target in zip(updates, target_updates):
        assert np.allclose(actual, target)
    for actual, target in zip(next_state.velocities, expected.velocities):
        assert np.allclose(actual, target)
    assert next_state.learning_rate == expected.learning_rate


def test_mlp_sgd_updates_match_sklearn_with_nesterov() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_optimizers import mlp_sgd_initialize_state, mlp_sgd_updates

    params = _params()
    grads = _grads()
    state = mlp_sgd_initialize_state(params, learning_rate_init=0.2, lr_schedule="constant", momentum=0.9, nesterov=True)
    updates, next_state = mlp_sgd_updates(grads, state)

    expected = SGDOptimizer(list(params), learning_rate_init=0.2, lr_schedule="constant", momentum=0.9, nesterov=True)
    target_updates = expected._get_updates(list(grads))

    for actual, target in zip(updates, target_updates):
        assert np.allclose(actual, target)
    for actual, target in zip(next_state.velocities, expected.velocities):
        assert np.allclose(actual, target)


def test_mlp_sgd_iteration_end_matches_sklearn_invscaling() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_optimizers import mlp_sgd_initialize_state, mlp_sgd_iteration_end

    params = _params()
    state = mlp_sgd_initialize_state(params, learning_rate_init=0.2, lr_schedule="invscaling", power_t=0.5)
    result = mlp_sgd_iteration_end(state, time_step=9)

    expected = SGDOptimizer(list(params), learning_rate_init=0.2, lr_schedule="invscaling", power_t=0.5)
    expected.iteration_ends(9)

    assert result.learning_rate == pytest.approx(expected.learning_rate)


def test_mlp_sgd_trigger_stopping_matches_sklearn_adaptive() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_optimizers import mlp_sgd_initialize_state, mlp_sgd_trigger_stopping

    params = _params()
    state = mlp_sgd_initialize_state(params, learning_rate_init=0.2, lr_schedule="adaptive")
    should_stop, next_state = mlp_sgd_trigger_stopping(state)

    expected = SGDOptimizer(list(params), learning_rate_init=0.2, lr_schedule="adaptive")
    target_should_stop = expected.trigger_stopping("msg", False)

    assert should_stop is target_should_stop
    assert next_state.learning_rate == pytest.approx(expected.learning_rate)

    tiny_state = mlp_sgd_initialize_state(params, learning_rate_init=0.2, lr_schedule="adaptive")
    tiny_state = tiny_state.__class__(**{**tiny_state.__dict__, "learning_rate": 1e-7})
    should_stop_tiny, next_tiny_state = mlp_sgd_trigger_stopping(tiny_state)
    expected_tiny = SGDOptimizer(list(params), learning_rate_init=0.2, lr_schedule="adaptive")
    expected_tiny.learning_rate = 1e-7
    target_should_stop_tiny = expected_tiny.trigger_stopping("msg", False)

    assert should_stop_tiny is target_should_stop_tiny
    assert next_tiny_state.learning_rate == pytest.approx(expected_tiny.learning_rate)


def test_mlp_adam_initialize_state_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_optimizers import mlp_adam_initialize_state

    params = _params()
    state = mlp_adam_initialize_state(params, learning_rate_init=0.01, beta_1=0.8, beta_2=0.9, epsilon=1e-7)
    expected = AdamOptimizer(list(params), learning_rate_init=0.01, beta_1=0.8, beta_2=0.9, epsilon=1e-7)

    assert state.learning_rate_init == expected.learning_rate_init
    assert state.learning_rate == expected.learning_rate
    assert state.beta_1 == expected.beta_1
    assert state.beta_2 == expected.beta_2
    assert state.epsilon == expected.epsilon
    assert state.t == expected.t
    for actual, target in zip(state.ms, expected.ms):
        assert np.array_equal(actual, target)
    for actual, target in zip(state.vs, expected.vs):
        assert np.array_equal(actual, target)


def test_mlp_adam_updates_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_optimizers import mlp_adam_initialize_state, mlp_adam_updates

    params = _params()
    grads = _grads()
    state = mlp_adam_initialize_state(params, learning_rate_init=0.01, beta_1=0.8, beta_2=0.9, epsilon=1e-8)
    updates, next_state = mlp_adam_updates(grads, state)

    expected = AdamOptimizer(list(params), learning_rate_init=0.01, beta_1=0.8, beta_2=0.9, epsilon=1e-8)
    target_updates = expected._get_updates(list(grads))

    for actual, target in zip(updates, target_updates):
        assert np.allclose(actual, target)
    assert next_state.t == expected.t
    assert next_state.learning_rate == pytest.approx(expected.learning_rate)
    for actual, target in zip(next_state.ms, expected.ms):
        assert np.allclose(actual, target)
    for actual, target in zip(next_state.vs, expected.vs):
        assert np.allclose(actual, target)


def test_mlp_optimizer_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_optimizers import (
        mlp_adam_initialize_state,
        mlp_adam_updates,
        mlp_sgd_initialize_state,
        mlp_sgd_iteration_end,
        mlp_sgd_updates,
    )

    with pytest.raises(ViolationError):
        mlp_sgd_initialize_state(tuple(), learning_rate_init=0.1)

    with pytest.raises(ViolationError):
        mlp_sgd_iteration_end(
            mlp_sgd_initialize_state(_params()),
            time_step=-1,
        )

    with pytest.raises(ViolationError):
        mlp_sgd_updates(
            (np.ones((2, 2), dtype=np.float64),),
            mlp_sgd_initialize_state(_params()),
        )

    with pytest.raises(ViolationError):
        mlp_adam_initialize_state(_params(), beta_1=1.1)

    with pytest.raises(ViolationError):
        mlp_adam_updates(
            (np.ones((2, 2), dtype=np.float64),),
            mlp_adam_initialize_state(_params()),
        )
