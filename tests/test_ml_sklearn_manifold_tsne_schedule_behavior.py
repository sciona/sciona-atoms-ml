from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy.sparse import csr_matrix


def test_tsne_schedule_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_schedule import (
        tsne_early_exaggeration_scale,
        tsne_early_exaggeration_unscale,
        tsne_gradient_descent_buffers,
        tsne_gradient_descent_compute_error,
        tsne_gradient_descent_convergence,
        tsne_stage_two_required,
    )

    assert callable(tsne_gradient_descent_buffers)
    assert callable(tsne_gradient_descent_compute_error)
    assert callable(tsne_gradient_descent_convergence)
    assert callable(tsne_early_exaggeration_scale)
    assert callable(tsne_early_exaggeration_unscale)
    assert callable(tsne_stage_two_required)


def test_gradient_descent_buffers_allocate_flattened_zero_initialized_state() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_schedule import tsne_gradient_descent_buffers

    p0 = np.array([0.2, -0.3, 0.4], dtype=np.float64)
    params, update, gains = tsne_gradient_descent_buffers(p0)

    assert np.allclose(params, p0)
    assert np.allclose(update, np.zeros_like(p0))
    assert np.allclose(gains, np.ones_like(p0))
    assert params.dtype == np.float64
    assert update.dtype == np.float64
    assert gains.dtype == np.float64


@pytest.mark.parametrize(
    ("iteration", "max_iter", "n_iter_check", "expected"),
    [
        (0, 5, 2, False),
        (1, 5, 2, True),
        (3, 5, 2, True),
        (4, 5, 2, True),
    ],
)
def test_gradient_descent_compute_error_matches_private_schedule_rule(
    iteration: int,
    max_iter: int,
    n_iter_check: int,
    expected: bool,
) -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_schedule import tsne_gradient_descent_compute_error

    actual = tsne_gradient_descent_compute_error(iteration, max_iter, n_iter_check=n_iter_check)

    reference = ((iteration + 1) % n_iter_check == 0) or (iteration == max_iter - 1)
    assert actual is expected
    assert actual is reference


def test_gradient_descent_convergence_updates_best_error_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_schedule import tsne_gradient_descent_convergence

    grad = np.array([3.0, 4.0], dtype=np.float64)
    best_error, best_iter, grad_norm, should_stop = tsne_gradient_descent_convergence(
        7.0,
        grad,
        5,
        8.0,
        3,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
    )

    assert best_error == pytest.approx(7.0)
    assert best_iter == 5
    assert grad_norm == pytest.approx(5.0)
    assert should_stop is False


def test_gradient_descent_convergence_stops_after_no_progress_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_schedule import tsne_gradient_descent_convergence

    grad = np.array([1.0, 0.0], dtype=np.float64)
    best_error, best_iter, grad_norm, should_stop = tsne_gradient_descent_convergence(
        9.0,
        grad,
        11,
        8.0,
        5,
        n_iter_without_progress=5,
        min_grad_norm=1e-7,
    )

    assert best_error == pytest.approx(8.0)
    assert best_iter == 5
    assert grad_norm == pytest.approx(1.0)
    assert should_stop is True


def test_gradient_descent_convergence_stops_when_gradient_norm_is_small() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_schedule import tsne_gradient_descent_convergence

    grad = np.array([1e-9, 0.0], dtype=np.float64)
    best_error, best_iter, grad_norm, should_stop = tsne_gradient_descent_convergence(
        7.0,
        grad,
        2,
        8.0,
        1,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
    )

    assert best_error == pytest.approx(7.0)
    assert best_iter == 2
    assert grad_norm <= 1e-7
    assert should_stop is True


def test_early_exaggeration_scale_and_unscale_match_private_schedule_dense() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_schedule import (
        tsne_early_exaggeration_scale,
        tsne_early_exaggeration_unscale,
    )

    probabilities = np.array([0.02, 0.03, 0.05], dtype=np.float64)
    early_exaggeration = 12.0

    scaled = tsne_early_exaggeration_scale(probabilities, early_exaggeration)
    restored = tsne_early_exaggeration_unscale(scaled, early_exaggeration)

    assert np.allclose(scaled, probabilities * early_exaggeration)
    assert np.allclose(restored, probabilities)


def test_early_exaggeration_scale_and_unscale_preserve_csr_storage() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_schedule import (
        tsne_early_exaggeration_scale,
        tsne_early_exaggeration_unscale,
    )

    probabilities = csr_matrix(
        np.array(
            [
                [0.0, 0.2, 0.0],
                [0.1, 0.0, 0.3],
                [0.0, 0.4, 0.0],
            ],
            dtype=np.float64,
        )
    )

    scaled = tsne_early_exaggeration_scale(probabilities, 4.0)
    restored = tsne_early_exaggeration_unscale(scaled, 4.0)

    assert isinstance(scaled, csr_matrix)
    assert isinstance(restored, csr_matrix)
    assert np.allclose(scaled.toarray(), probabilities.toarray() * 4.0)
    assert np.allclose(restored.toarray(), probabilities.toarray())


@pytest.mark.parametrize(
    ("iteration", "exploration_max_iter", "max_iter", "expected"),
    [
        (249, 250, 250, True),
        (100, 250, 1000, True),
        (250, 250, 250, False),
    ],
)
def test_stage_two_required_matches_private_control_flow_edges(
    iteration: int,
    exploration_max_iter: int,
    max_iter: int,
    expected: bool,
) -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_schedule import tsne_stage_two_required

    actual = tsne_stage_two_required(iteration, exploration_max_iter, max_iter)
    reference = (iteration < exploration_max_iter) or ((max_iter - exploration_max_iter) > 0)

    assert actual is expected
    assert actual is reference


def test_tsne_schedule_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_schedule import (
        tsne_early_exaggeration_scale,
        tsne_gradient_descent_buffers,
        tsne_gradient_descent_compute_error,
        tsne_gradient_descent_convergence,
        tsne_stage_two_required,
    )

    with pytest.raises(ViolationError):
        tsne_gradient_descent_buffers(np.array([[1.0, 2.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        tsne_gradient_descent_compute_error(5, 5)

    with pytest.raises(ViolationError):
        tsne_gradient_descent_convergence(1.0, np.array([1.0], dtype=np.float64), 1, 2.0, 3)

    with pytest.raises(ViolationError):
        tsne_early_exaggeration_scale(np.array([-0.1, 0.2], dtype=np.float64), 12.0)

    with pytest.raises(ViolationError):
        tsne_stage_two_required(1, 250, 200)
