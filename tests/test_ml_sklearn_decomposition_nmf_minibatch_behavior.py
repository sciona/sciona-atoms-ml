from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import MiniBatchNMF


def test_nmf_minibatch_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf_minibatch import (
        nmf_minibatch_batch_size,
        nmf_minibatch_ewa_cost,
        nmf_minibatch_h_change_converged,
        nmf_minibatch_improvement_state,
        nmf_minibatch_mm_gamma,
        nmf_minibatch_rho,
        nmf_minibatch_transform_max_iter,
    )

    assert callable(nmf_minibatch_batch_size)
    assert callable(nmf_minibatch_rho)
    assert callable(nmf_minibatch_mm_gamma)
    assert callable(nmf_minibatch_transform_max_iter)
    assert callable(nmf_minibatch_ewa_cost)
    assert callable(nmf_minibatch_h_change_converged)
    assert callable(nmf_minibatch_improvement_state)


@pytest.mark.parametrize(
    ("batch_size", "forget_factor", "beta_loss", "max_iter", "transform_max_iter"),
    [
        (1024, 0.7, "frobenius", 200, None),
        (4, 0.5, 0.5, 40, 17),
        (2, 1.0, 3.5, 25, None),
    ],
)
def test_nmf_minibatch_parameter_helpers_match_sklearn_check_params(
    batch_size: int,
    forget_factor: float,
    beta_loss: float | str,
    max_iter: int,
    transform_max_iter: int | None,
) -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf_minibatch import (
        nmf_minibatch_batch_size,
        nmf_minibatch_mm_gamma,
        nmf_minibatch_rho,
        nmf_minibatch_transform_max_iter,
    )

    X = np.array(
        [
            [1.0, 0.5, 2.0],
            [0.0, 1.5, 1.0],
            [2.0, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    model = MiniBatchNMF(
        n_components=2,
        batch_size=batch_size,
        forget_factor=forget_factor,
        beta_loss=beta_loss,
        max_iter=max_iter,
        transform_max_iter=transform_max_iter,
        random_state=0,
    )
    model._check_params(X)

    assert nmf_minibatch_batch_size(batch_size, X.shape[0]) == model._batch_size
    assert np.isclose(
        nmf_minibatch_rho(forget_factor, model._batch_size, X.shape[0]),
        model._rho,
    )
    assert np.isclose(nmf_minibatch_mm_gamma(model._beta_loss), model._gamma)
    assert (
        nmf_minibatch_transform_max_iter(max_iter, transform_max_iter)
        == model._transform_max_iter
    )


def test_nmf_minibatch_convergence_helpers_match_sklearn_second_step() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf_minibatch import (
        nmf_minibatch_batch_size,
        nmf_minibatch_ewa_cost,
        nmf_minibatch_h_change_converged,
        nmf_minibatch_improvement_state,
    )

    X = np.array([[1.0, 0.0], [0.3, 2.0], [1.4, 0.8]], dtype=np.float64)
    H_buffer = np.array([[0.8, 0.1], [0.2, 0.7]], dtype=np.float64)
    H = np.array([[0.82, 0.11], [0.21, 0.69]], dtype=np.float64)
    batch_cost = 1.75

    model = MiniBatchNMF(n_components=2, tol=1e-4, max_no_improvement=3, random_state=0)
    model._ewa_cost = 1.5
    model._ewa_cost_min = 1.45
    model._no_improvement = 1
    stopped = model._minibatch_convergence(
        X=X,
        batch_cost=batch_cost,
        H=H,
        H_buffer=H_buffer,
        n_samples=11,
        step=1,
        n_steps=5,
    )

    resolved_batch_size = nmf_minibatch_batch_size(X.shape[0], X.shape[0])
    expected_ewa = nmf_minibatch_ewa_cost(1.5, batch_cost, resolved_batch_size, 11)
    expected_h_stop = nmf_minibatch_h_change_converged(H, H_buffer, model.tol)
    expected_min, expected_count, expected_stop = nmf_minibatch_improvement_state(
        expected_ewa,
        1.45,
        1,
        model.max_no_improvement,
    )

    assert np.isclose(model._ewa_cost, expected_ewa)
    assert expected_h_stop is False
    assert model._ewa_cost_min == pytest.approx(expected_min)
    assert model._no_improvement == expected_count
    assert stopped is expected_stop


def test_nmf_minibatch_convergence_helpers_match_small_h_change_stop() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf_minibatch import (
        nmf_minibatch_batch_size,
        nmf_minibatch_ewa_cost,
        nmf_minibatch_h_change_converged,
    )

    X = np.array([[0.4, 0.9], [1.2, 0.7]], dtype=np.float64)
    H_buffer = np.array([[1.0, 0.5], [0.4, 0.2]], dtype=np.float64)
    H = H_buffer + 1e-7
    batch_cost = 0.9

    model = MiniBatchNMF(n_components=2, tol=1e-4, max_no_improvement=5, random_state=0)
    model._ewa_cost = None
    model._ewa_cost_min = None
    model._no_improvement = 0
    stopped = model._minibatch_convergence(
        X=X,
        batch_cost=batch_cost,
        H=H,
        H_buffer=H_buffer,
        n_samples=20,
        step=1,
        n_steps=4,
    )

    resolved_batch_size = nmf_minibatch_batch_size(X.shape[0], X.shape[0])
    expected_ewa = nmf_minibatch_ewa_cost(None, batch_cost, resolved_batch_size, 20)

    assert np.isclose(model._ewa_cost, expected_ewa)
    assert nmf_minibatch_h_change_converged(H, H_buffer, model.tol) is True
    assert stopped is True


def test_nmf_minibatch_convergence_helpers_match_first_step_short_circuit() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf_minibatch import nmf_minibatch_h_change_converged

    X = np.array([[1.0, 2.0], [0.4, 0.6]], dtype=np.float64)
    H_buffer = np.array([[0.4, 0.5], [0.2, 0.1]], dtype=np.float64)
    H = H_buffer.copy()

    model = MiniBatchNMF(n_components=2, tol=1e-4, max_no_improvement=2, random_state=0)
    model._ewa_cost = 2.0
    model._ewa_cost_min = 1.5
    model._no_improvement = 1
    stopped = model._minibatch_convergence(
        X=X,
        batch_cost=1.1,
        H=H,
        H_buffer=H_buffer,
        n_samples=10,
        step=0,
        n_steps=3,
    )

    assert nmf_minibatch_h_change_converged(H, H_buffer, model.tol) is True
    assert stopped is False
    assert model._ewa_cost == 2.0
    assert model._ewa_cost_min == 1.5
    assert model._no_improvement == 1


def test_nmf_minibatch_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf_minibatch import (
        nmf_minibatch_batch_size,
        nmf_minibatch_ewa_cost,
        nmf_minibatch_h_change_converged,
        nmf_minibatch_improvement_state,
        nmf_minibatch_mm_gamma,
        nmf_minibatch_rho,
        nmf_minibatch_transform_max_iter,
    )

    H = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(Exception):
        nmf_minibatch_batch_size(0, 3)
    with pytest.raises(Exception):
        nmf_minibatch_rho(1.2, 2, 3)
    with pytest.raises(Exception):
        nmf_minibatch_mm_gamma(float("nan"))
    with pytest.raises(Exception):
        nmf_minibatch_transform_max_iter(5, 0)
    with pytest.raises(Exception):
        nmf_minibatch_ewa_cost(1.0, float("inf"), 2, 3)
    with pytest.raises(Exception):
        nmf_minibatch_h_change_converged(H, np.ones((2, 3), dtype=np.float64), 1e-4)
    with pytest.raises(Exception):
        nmf_minibatch_improvement_state(1.0, 0.5, -1, 2)
