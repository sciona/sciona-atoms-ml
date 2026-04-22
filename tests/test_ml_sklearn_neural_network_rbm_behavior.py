from __future__ import annotations

import numpy as np
import pytest
from sklearn.neural_network import BernoulliRBM


def _binary_data() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def _sklearn_from_state(state, *, random_state: int | None = None) -> BernoulliRBM:
    model = BernoulliRBM(
        n_components=state.n_components,
        learning_rate=state.learning_rate,
        batch_size=state.batch_size,
        n_iter=state.n_iter,
        random_state=random_state,
    )
    model.components_ = np.asarray(state.components, dtype=np.float64, order="F").copy()
    model.intercept_hidden_ = state.intercept_hidden.copy()
    model.intercept_visible_ = state.intercept_visible.copy()
    model.h_samples_ = state.h_samples.copy()
    model.n_features_in_ = state.n_features_in
    model._n_features_out = state.n_components
    model.random_state_ = np.random.RandomState(random_state)
    return model


def test_bernoulli_rbm_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neural_network import (
        BernoulliRBMState,
        bernoulli_rbm_fit,
        bernoulli_rbm_free_energy,
        bernoulli_rbm_gibbs,
        bernoulli_rbm_mean_hiddens,
        bernoulli_rbm_partial_fit,
        bernoulli_rbm_sample_hiddens,
        bernoulli_rbm_sample_visibles,
        bernoulli_rbm_score_samples,
    )

    assert BernoulliRBMState is not None
    assert callable(bernoulli_rbm_fit)
    assert callable(bernoulli_rbm_free_energy)
    assert callable(bernoulli_rbm_gibbs)
    assert callable(bernoulli_rbm_mean_hiddens)
    assert callable(bernoulli_rbm_partial_fit)
    assert callable(bernoulli_rbm_sample_hiddens)
    assert callable(bernoulli_rbm_sample_visibles)
    assert callable(bernoulli_rbm_score_samples)


def test_bernoulli_rbm_fit_and_scores_match_sklearn_dense_binary() -> None:
    from sciona.atoms.ml.sklearn.neural_network import (
        bernoulli_rbm_fit,
        bernoulli_rbm_free_energy,
        bernoulli_rbm_mean_hiddens,
        bernoulli_rbm_score_samples,
    )

    X = _binary_data()
    expected = BernoulliRBM(
        n_components=3,
        learning_rate=0.05,
        batch_size=4,
        n_iter=3,
        random_state=11,
    ).fit(X)
    state = bernoulli_rbm_fit(
        X,
        n_components=3,
        learning_rate=0.05,
        batch_size=4,
        n_iter=3,
        random_state=11,
    )

    assert np.allclose(state.components, expected.components_)
    assert np.allclose(state.intercept_hidden, expected.intercept_hidden_)
    assert np.allclose(state.intercept_visible, expected.intercept_visible_)
    assert np.allclose(state.h_samples, expected.h_samples_)
    assert np.allclose(bernoulli_rbm_mean_hiddens(X, state), expected._mean_hiddens(X))
    assert np.allclose(bernoulli_rbm_free_energy(X, state), expected._free_energy(X))
    assert np.allclose(bernoulli_rbm_score_samples(X, state, random_state=11), expected.score_samples(X))


def test_bernoulli_rbm_sampling_atoms_match_sklearn_private_helpers() -> None:
    from sciona.atoms.ml.sklearn.neural_network import (
        bernoulli_rbm_fit,
        bernoulli_rbm_gibbs,
        bernoulli_rbm_sample_hiddens,
        bernoulli_rbm_sample_visibles,
    )

    X = _binary_data()
    state = bernoulli_rbm_fit(X, n_components=2, batch_size=3, n_iter=1, random_state=5)
    model = _sklearn_from_state(state, random_state=19)
    hidden = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    assert np.array_equal(bernoulli_rbm_sample_hiddens(X[:2], state, random_state=19), model._sample_hiddens(X[:2], np.random.RandomState(19)))
    assert np.array_equal(bernoulli_rbm_sample_visibles(hidden, state, random_state=19), model._sample_visibles(hidden, np.random.RandomState(19)))
    assert np.array_equal(bernoulli_rbm_gibbs(X[:2], state, random_state=19), model.gibbs(X[:2]))


def test_bernoulli_rbm_partial_fit_matches_sklearn_update() -> None:
    from sciona.atoms.ml.sklearn.neural_network import bernoulli_rbm_fit, bernoulli_rbm_partial_fit

    X = _binary_data()
    initial = bernoulli_rbm_fit(X, n_components=3, learning_rate=0.05, batch_size=4, n_iter=0, random_state=7)
    expected = _sklearn_from_state(initial, random_state=13)
    expected.partial_fit(X[:4])
    updated = bernoulli_rbm_partial_fit(X[:4], initial, random_state=13)

    assert np.allclose(updated.components, expected.components_)
    assert np.allclose(updated.intercept_hidden, expected.intercept_hidden_)
    assert np.allclose(updated.intercept_visible, expected.intercept_visible_)
    assert np.allclose(updated.h_samples, expected.h_samples_)


def test_bernoulli_rbm_rejects_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.neural_network import bernoulli_rbm_fit, bernoulli_rbm_mean_hiddens

    X = _binary_data()
    state = bernoulli_rbm_fit(X, n_components=2, batch_size=3, n_iter=1, random_state=3)
    with pytest.raises(Exception):
        bernoulli_rbm_fit(np.array([[0.0, 2.0]], dtype=np.float64), n_components=2)
    with pytest.raises(Exception):
        bernoulli_rbm_fit(X, n_components=0)
    with pytest.raises(Exception):
        bernoulli_rbm_mean_hiddens(X[:, :2], state)
