from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import MiniBatchDictionaryLearning


def test_dictionary_learning_minibatch_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch import (
        dictionary_learning_minibatch_batch_size,
        dictionary_learning_minibatch_component_count,
        dictionary_learning_minibatch_dictionary_change_converged,
        dictionary_learning_minibatch_ewa_cost,
        dictionary_learning_minibatch_fit_algorithm,
        dictionary_learning_minibatch_improvement_state,
        dictionary_learning_minibatch_inner_stats,
        dictionary_learning_minibatch_monitoring_started,
        dictionary_learning_minibatch_stats_decay,
    )

    assert callable(dictionary_learning_minibatch_component_count)
    assert callable(dictionary_learning_minibatch_fit_algorithm)
    assert callable(dictionary_learning_minibatch_batch_size)
    assert callable(dictionary_learning_minibatch_stats_decay)
    assert callable(dictionary_learning_minibatch_inner_stats)
    assert callable(dictionary_learning_minibatch_monitoring_started)
    assert callable(dictionary_learning_minibatch_ewa_cost)
    assert callable(dictionary_learning_minibatch_dictionary_change_converged)
    assert callable(dictionary_learning_minibatch_improvement_state)


@pytest.mark.parametrize(
    ("n_components", "fit_algorithm", "positive_code", "batch_size"),
    [
        (None, "cd", False, 50),
        (3, "cd", True, 2),
        (4, "lars", False, 8),
    ],
)
def test_dictionary_learning_minibatch_parameter_helpers_match_sklearn_check_params(
    n_components: int | None,
    fit_algorithm: str,
    positive_code: bool,
    batch_size: int,
) -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch import (
        dictionary_learning_minibatch_batch_size,
        dictionary_learning_minibatch_component_count,
        dictionary_learning_minibatch_fit_algorithm,
    )

    X = np.array(
        [
            [1.0, 0.2, 0.3],
            [0.1, 0.5, 0.9],
            [0.4, 0.8, 0.6],
        ],
        dtype=np.float64,
    )
    model = MiniBatchDictionaryLearning(
        n_components=n_components,
        fit_algorithm=fit_algorithm,
        positive_code=positive_code,
        batch_size=batch_size,
        random_state=0,
    )
    model._check_params(X)

    assert dictionary_learning_minibatch_component_count(n_components, X.shape[1]) == model._n_components
    assert dictionary_learning_minibatch_fit_algorithm(fit_algorithm, positive_code) == model._fit_algorithm
    assert dictionary_learning_minibatch_batch_size(batch_size, X.shape[0]) == model._batch_size


def test_dictionary_learning_minibatch_fit_algorithm_rejects_positive_lars() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch import dictionary_learning_minibatch_fit_algorithm

    with pytest.raises(Exception):
        dictionary_learning_minibatch_fit_algorithm("lars", True)


def test_dictionary_learning_minibatch_inner_stats_helpers_match_sklearn_update() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch import (
        dictionary_learning_minibatch_inner_stats,
        dictionary_learning_minibatch_stats_decay,
    )

    X_batch = np.array([[1.0, 0.0], [0.5, 1.5]], dtype=np.float64)
    code = np.array([[0.2, 0.3, 0.1], [0.4, 0.1, 0.6]], dtype=np.float64)
    batch_size = X_batch.shape[0]
    step = 3

    model = MiniBatchDictionaryLearning(n_components=3, random_state=0)
    model._A = np.array([[1.0, 0.2, 0.0], [0.2, 0.8, 0.1], [0.0, 0.1, 0.9]], dtype=np.float64)
    model._B = np.array([[0.5, 0.3, 0.7], [0.2, 0.6, 0.1]], dtype=np.float64)
    expected_A = model._A.copy()
    expected_B = model._B.copy()
    model._update_inner_stats(X_batch, code, batch_size, step)

    decay = dictionary_learning_minibatch_stats_decay(batch_size, step)
    actual_A, actual_B = dictionary_learning_minibatch_inner_stats(
        expected_A,
        expected_B,
        X_batch,
        code,
        batch_size=batch_size,
        step=step,
    )

    assert decay == pytest.approx((batch_size**2 + step + 1 - batch_size + 1 - batch_size) / (batch_size**2 + step + 1 - batch_size + 1))
    assert np.allclose(actual_A, model._A)
    assert np.allclose(actual_B, model._B)


def test_dictionary_learning_minibatch_convergence_helpers_match_warmup_short_circuit() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch import (
        dictionary_learning_minibatch_monitoring_started,
    )

    X_batch = np.array([[1.0, 0.2], [0.3, 0.4]], dtype=np.float64)
    new_dict = np.array([[0.5, 0.1], [0.2, 0.7]], dtype=np.float64)
    old_dict = new_dict.copy()

    model = MiniBatchDictionaryLearning(n_components=2, tol=1e-4, max_no_improvement=2, random_state=0)
    model.verbose = False
    model._n_components = 2
    model._ewa_cost = 1.7
    model._ewa_cost_min = 1.5
    model._no_improvement = 1

    stopped = model._check_convergence(
        X_batch,
        0.9,
        new_dict,
        old_dict,
        n_samples=10,
        step=0,
        n_steps=3,
    )

    assert dictionary_learning_minibatch_monitoring_started(0, 10, X_batch.shape[0]) is False
    assert stopped is False
    assert model._ewa_cost == 1.7
    assert model._ewa_cost_min == 1.5
    assert model._no_improvement == 1


def test_dictionary_learning_minibatch_convergence_helpers_match_dictionary_change_stop() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch import (
        dictionary_learning_minibatch_dictionary_change_converged,
        dictionary_learning_minibatch_ewa_cost,
        dictionary_learning_minibatch_monitoring_started,
    )

    X_batch = np.array([[1.0, 0.0], [0.3, 0.8]], dtype=np.float64)
    old_dict = np.array([[0.5, 0.1], [0.2, 0.7]], dtype=np.float64)
    new_dict = old_dict + 1e-7
    batch_cost = 0.8

    model = MiniBatchDictionaryLearning(n_components=2, tol=1e-4, max_no_improvement=5, random_state=0)
    model.verbose = False
    model._n_components = 2
    model._ewa_cost = None
    model._ewa_cost_min = None
    model._no_improvement = 0

    stopped = model._check_convergence(
        X_batch,
        batch_cost,
        new_dict,
        old_dict,
        n_samples=2,
        step=1,
        n_steps=4,
    )

    assert dictionary_learning_minibatch_monitoring_started(1, 2, X_batch.shape[0]) is True
    expected_ewa = dictionary_learning_minibatch_ewa_cost(None, batch_cost, X_batch.shape[0], 2)
    assert model._ewa_cost == pytest.approx(expected_ewa)
    assert dictionary_learning_minibatch_dictionary_change_converged(
        new_dict,
        old_dict,
        n_components=2,
        tol=model.tol,
    ) is True
    assert stopped is True


def test_dictionary_learning_minibatch_convergence_helpers_match_no_improvement_stop() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch import (
        dictionary_learning_minibatch_dictionary_change_converged,
        dictionary_learning_minibatch_ewa_cost,
        dictionary_learning_minibatch_improvement_state,
        dictionary_learning_minibatch_monitoring_started,
    )

    X_batch = np.array([[1.0, 0.0], [0.3, 0.8]], dtype=np.float64)
    old_dict = np.array([[0.5, 0.1], [0.2, 0.7]], dtype=np.float64)
    new_dict = np.array([[0.9, 0.3], [0.1, 0.6]], dtype=np.float64)
    batch_cost = 1.2

    model = MiniBatchDictionaryLearning(n_components=2, tol=1e-8, max_no_improvement=2, random_state=0)
    model.verbose = False
    model._n_components = 2
    model._ewa_cost = 1.1
    model._ewa_cost_min = 1.0
    model._no_improvement = 1

    stopped = model._check_convergence(
        X_batch,
        batch_cost,
        new_dict,
        old_dict,
        n_samples=2,
        step=1,
        n_steps=4,
    )

    assert dictionary_learning_minibatch_monitoring_started(1, 2, X_batch.shape[0]) is True
    expected_ewa = dictionary_learning_minibatch_ewa_cost(1.1, batch_cost, X_batch.shape[0], 2)
    expected_min, expected_count, expected_stop = dictionary_learning_minibatch_improvement_state(
        expected_ewa,
        1.0,
        1,
        2,
    )
    assert dictionary_learning_minibatch_dictionary_change_converged(
        new_dict,
        old_dict,
        n_components=2,
        tol=model.tol,
    ) is False
    assert model._ewa_cost == pytest.approx(expected_ewa)
    assert model._ewa_cost_min == pytest.approx(expected_min)
    assert model._no_improvement == expected_count
    assert stopped is expected_stop


def test_dictionary_learning_minibatch_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch import (
        dictionary_learning_minibatch_batch_size,
        dictionary_learning_minibatch_component_count,
        dictionary_learning_minibatch_dictionary_change_converged,
        dictionary_learning_minibatch_ewa_cost,
        dictionary_learning_minibatch_fit_algorithm,
        dictionary_learning_minibatch_improvement_state,
        dictionary_learning_minibatch_inner_stats,
        dictionary_learning_minibatch_monitoring_started,
        dictionary_learning_minibatch_stats_decay,
    )

    A = np.eye(2, dtype=np.float64)
    B = np.ones((3, 2), dtype=np.float64)
    X_batch = np.ones((2, 3), dtype=np.float64)
    code = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(Exception):
        dictionary_learning_minibatch_component_count(0, 3)
    with pytest.raises(Exception):
        dictionary_learning_minibatch_fit_algorithm("omp", False)
    with pytest.raises(Exception):
        dictionary_learning_minibatch_batch_size(0, 3)
    with pytest.raises(Exception):
        dictionary_learning_minibatch_stats_decay(2, -1)
    with pytest.raises(Exception):
        dictionary_learning_minibatch_inner_stats(A, B, X_batch, code, batch_size=3, step=0)
    with pytest.raises(Exception):
        dictionary_learning_minibatch_monitoring_started(-1, 5, 2)
    with pytest.raises(Exception):
        dictionary_learning_minibatch_ewa_cost(1.0, float("inf"), 2, 3)
    with pytest.raises(Exception):
        dictionary_learning_minibatch_dictionary_change_converged(np.ones((2, 2)), np.ones((2, 3)), n_components=2, tol=1e-3)
    with pytest.raises(Exception):
        dictionary_learning_minibatch_improvement_state(1.0, 0.5, -1, 2)
