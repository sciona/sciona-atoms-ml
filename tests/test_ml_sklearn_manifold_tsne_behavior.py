from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import _utils
from sklearn.manifold._t_sne import _gradient_descent, _joint_probabilities, _kl_divergence


def _embedding_fixture() -> tuple[np.ndarray, np.ndarray, int, int, int]:
    X_embedded = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [0.3, 1.1],
            [1.2, 1.0],
        ],
        dtype=np.float64,
    )
    params = X_embedded.ravel()
    distances = squareform(pdist(X_embedded, "sqeuclidean"))
    expected_p = _joint_probabilities(distances.copy(), 2.0, 0)
    return params, expected_p, 1, X_embedded.shape[0], X_embedded.shape[1]


def test_tsne_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne import (
        tsne_exact_joint_probabilities,
        tsne_exact_kl_divergence,
        tsne_gradient_descent_update,
    )

    assert callable(tsne_exact_joint_probabilities)
    assert callable(tsne_exact_kl_divergence)
    assert callable(tsne_gradient_descent_update)


def test_exact_joint_probabilities_matches_sklearn_post_binary_search() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne import tsne_exact_joint_probabilities

    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.1],
            [0.2, 0.8],
            [0.9, 1.0],
        ],
        dtype=np.float64,
    )
    distances = squareform(pdist(X, "sqeuclidean"))
    conditional = _utils._binary_search_perplexity(distances.astype(np.float32), 2.0, 0)

    actual = tsne_exact_joint_probabilities(conditional.astype(np.float64))
    expected = _joint_probabilities(distances.copy(), 2.0, 0)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)
    assert np.isclose(actual.sum(), 0.5)


def test_exact_kl_divergence_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne import tsne_exact_kl_divergence

    params, P, degrees_of_freedom, n_samples, n_components = _embedding_fixture()

    actual_error, actual_grad = tsne_exact_kl_divergence(
        params,
        P,
        degrees_of_freedom,
        n_samples,
        n_components,
    )
    expected_error, expected_grad = _kl_divergence(
        params.copy(),
        P.copy(),
        degrees_of_freedom,
        n_samples,
        n_components,
    )

    assert np.isclose(actual_error, expected_error)
    assert np.allclose(actual_grad, expected_grad)


def test_exact_kl_divergence_can_skip_error_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne import tsne_exact_kl_divergence

    params, P, degrees_of_freedom, n_samples, n_components = _embedding_fixture()
    actual_error, actual_grad = tsne_exact_kl_divergence(
        params,
        P,
        degrees_of_freedom,
        n_samples,
        n_components,
        compute_error=False,
    )
    expected_error, expected_grad = _kl_divergence(
        params.copy(),
        P.copy(),
        degrees_of_freedom,
        n_samples,
        n_components,
        compute_error=False,
    )

    assert np.isnan(actual_error)
    assert np.isnan(expected_error)
    assert np.allclose(actual_grad, expected_grad)


def test_gradient_descent_update_matches_first_sklearn_iteration() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne import tsne_gradient_descent_update

    p0 = np.array([0.2, -0.3, 0.4], dtype=np.float64)
    grad = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    def objective(params: np.ndarray, *, compute_error: bool = True) -> tuple[float, np.ndarray]:
        del params, compute_error
        return 7.0, grad.copy()

    actual_p, actual_update, actual_gains = tsne_gradient_descent_update(
        p0,
        np.zeros_like(p0),
        np.ones_like(p0),
        grad,
        momentum=0.5,
        learning_rate=10.0,
        min_gain=0.01,
    )
    expected_p, expected_error, expected_iter = _gradient_descent(
        objective,
        p0,
        0,
        1,
        n_iter_check=1,
        momentum=0.5,
        learning_rate=10.0,
        min_gain=0.01,
        min_grad_norm=0.0,
    )

    assert expected_error == 7.0
    assert expected_iter == 0
    assert np.allclose(actual_p, expected_p)
    assert np.allclose(actual_update, actual_p - p0)
    assert np.allclose(actual_gains, np.full_like(p0, 0.8))


def test_gradient_descent_update_applies_adaptive_gain_sign_rule() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne import tsne_gradient_descent_update

    p = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    update = np.array([0.2, -0.4, 0.1], dtype=np.float64)
    gains = np.array([0.5, 0.5, 0.005], dtype=np.float64)
    grad = np.array([-0.3, -0.2, 0.4], dtype=np.float64)

    new_p, new_update, new_gains = tsne_gradient_descent_update(
        p,
        update,
        gains,
        grad,
        momentum=0.25,
        learning_rate=2.0,
        min_gain=0.01,
    )

    assert np.allclose(new_gains, np.array([0.7, 0.4, 0.01]))
    assert np.allclose(new_update, 0.25 * update - 2.0 * grad * new_gains)
    assert np.allclose(new_p, p + new_update)


def test_contracts_reject_invalid_tsne_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne import (
        tsne_exact_joint_probabilities,
        tsne_exact_kl_divergence,
        tsne_gradient_descent_update,
    )

    params, P, degrees_of_freedom, n_samples, n_components = _embedding_fixture()

    with pytest.raises(ViolationError):
        tsne_exact_joint_probabilities(np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        tsne_exact_kl_divergence(params[:-1], P, degrees_of_freedom, n_samples, n_components)

    with pytest.raises(ViolationError):
        tsne_exact_kl_divergence(params, P, degrees_of_freedom, n_samples, n_components, skip_num_points=1)

    with pytest.raises(ViolationError):
        tsne_gradient_descent_update(params, np.zeros_like(params), np.ones(2, dtype=np.float64), params)

    with pytest.raises(ViolationError):
        tsne_gradient_descent_update(params, np.zeros_like(params), np.ones_like(params), params, momentum=1.0)
