"""Dense Bernoulli restricted Boltzmann machine atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from sciona.ghost.registry import register_atom

from .state_models import BernoulliRBMState
from .witnesses import (
    witness_bernoulli_rbm_fit,
    witness_bernoulli_rbm_free_energy,
    witness_bernoulli_rbm_gibbs,
    witness_bernoulli_rbm_mean_hiddens,
    witness_bernoulli_rbm_partial_fit,
    witness_bernoulli_rbm_sample_hiddens,
    witness_bernoulli_rbm_sample_visibles,
    witness_bernoulli_rbm_score_samples,
)


def _visible_matrix(X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
    )


def _hidden_matrix(H: NDArray[np.float64], state: BernoulliRBMState) -> bool:
    try:
        values = np.asarray(H, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        _state_valid(state)
        and values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] == state.n_components
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
    )


def _state_valid(state: BernoulliRBMState) -> bool:
    return bool(
        state.n_features_in >= 1
        and state.n_components >= 1
        and state.batch_size >= 1
        and state.learning_rate > 0.0
        and state.n_iter >= 0
        and state.components.shape == (state.n_components, state.n_features_in)
        and state.intercept_hidden.shape == (state.n_components,)
        and state.intercept_visible.shape == (state.n_features_in,)
        and state.h_samples.shape == (state.batch_size, state.n_components)
        and np.all(np.isfinite(state.components))
        and np.all(np.isfinite(state.intercept_hidden))
        and np.all(np.isfinite(state.intercept_visible))
        and np.all(np.isfinite(state.h_samples))
        and np.all(state.h_samples >= 0.0)
        and np.all(state.h_samples <= 1.0)
    )


def _matrix_against_state(X: NDArray[np.float64], state: BernoulliRBMState) -> bool:
    return bool(_visible_matrix(X) and _state_valid(state) and np.asarray(X).shape[1] == state.n_features_in)


def _fit_parameters_valid(
    X: NDArray[np.float64],
    n_components: int,
    learning_rate: float,
    batch_size: int,
    n_iter: int,
) -> bool:
    return bool(
        _visible_matrix(X)
        and n_components >= 1
        and learning_rate > 0.0
        and np.isfinite(learning_rate)
        and batch_size >= 1
        and n_iter >= 0
    )


def _probability_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: BernoulliRBMState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (np.asarray(X).shape[0], state.n_components)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
    )


def _binary_matrix_result_valid(result: NDArray[np.float64], shape: tuple[int, int]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == shape and np.all(np.logical_or(values == 0.0, values == 1.0)))


def _free_energy_result_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (np.asarray(X).shape[0],) and np.all(np.isfinite(values)))


def _fit_result_valid(
    result: BernoulliRBMState,
    X: NDArray[np.float64],
    n_components: int,
    learning_rate: float,
    batch_size: int,
    n_iter: int,
) -> bool:
    return bool(
        _state_valid(result)
        and result.n_features_in == np.asarray(X).shape[1]
        and result.n_components == n_components
        and np.isclose(result.learning_rate, learning_rate)
        and result.batch_size == batch_size
        and result.n_iter == n_iter
    )


def _rng(random_state: int | None) -> np.random.RandomState:
    return np.random.RandomState(random_state)


def _fit_one_batch(
    X: NDArray[np.float64],
    state: BernoulliRBMState,
    rng: np.random.RandomState,
) -> BernoulliRBMState:
    h_pos = bernoulli_rbm_mean_hiddens(X, state)
    v_neg = _sample_visibles_with_rng(state.h_samples, state, rng)
    h_neg = bernoulli_rbm_mean_hiddens(v_neg, state)

    lr = float(state.learning_rate) / np.asarray(X).shape[0]
    update = np.dot(np.asarray(X, dtype=np.float64).T, h_pos).T
    update -= np.dot(h_neg.T, v_neg)
    components = state.components + lr * update
    intercept_hidden = state.intercept_hidden + lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
    intercept_visible = state.intercept_visible + lr * (np.asarray(X, dtype=np.float64).sum(axis=0) - v_neg.sum(axis=0))

    sampled_h = np.array(h_neg, copy=True)
    sampled_h[rng.uniform(size=sampled_h.shape) < sampled_h] = 1.0
    h_samples = np.floor(sampled_h, out=sampled_h)
    return BernoulliRBMState(
        components=np.asarray(components, dtype=np.float64, order="F"),
        intercept_hidden=np.asarray(intercept_hidden, dtype=np.float64),
        intercept_visible=np.asarray(intercept_visible, dtype=np.float64),
        h_samples=np.asarray(h_samples, dtype=np.float64),
        learning_rate=state.learning_rate,
        batch_size=state.batch_size,
        n_iter=state.n_iter,
        n_features_in=state.n_features_in,
        n_components=state.n_components,
    )


def _sample_hiddens_with_rng(
    X: NDArray[np.float64],
    state: BernoulliRBMState,
    rng: np.random.RandomState,
) -> NDArray[np.float64]:
    probabilities = bernoulli_rbm_mean_hiddens(X, state)
    return (rng.uniform(size=probabilities.shape) < probabilities).astype(np.float64)


def _sample_visibles_with_rng(
    H: NDArray[np.float64],
    state: BernoulliRBMState,
    rng: np.random.RandomState,
) -> NDArray[np.float64]:
    probabilities = np.dot(np.asarray(H, dtype=np.float64), state.components)
    probabilities += state.intercept_visible
    expit(probabilities, out=probabilities)
    return (rng.uniform(size=probabilities.shape) < probabilities).astype(np.float64)


@register_atom(witness_bernoulli_rbm_mean_hiddens)
@icontract.require(lambda X, state: _matrix_against_state(X, state), "visible samples must match fitted Bernoulli RBM state")
@icontract.ensure(lambda result, X, state: _probability_result_valid(result, X, state), "hidden means must be probabilities for each component")
def bernoulli_rbm_mean_hiddens(X: NDArray[np.float64], state: BernoulliRBMState) -> NDArray[np.float64]:
    """Compute hidden-unit activation probabilities for dense visible samples."""
    logits = np.dot(np.asarray(X, dtype=np.float64), state.components.T)
    logits += state.intercept_hidden
    return expit(logits, out=logits)


@register_atom(witness_bernoulli_rbm_sample_hiddens)
@icontract.require(lambda X, state, random_state: _matrix_against_state(X, state), "visible samples must match fitted Bernoulli RBM state")
@icontract.ensure(lambda result, X, state: _binary_matrix_result_valid(result, (np.asarray(X).shape[0], state.n_components)), "hidden samples must be binary")
def bernoulli_rbm_sample_hiddens(
    X: NDArray[np.float64],
    state: BernoulliRBMState,
    *,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Draw hidden Bernoulli states from visible activation probabilities."""
    return _sample_hiddens_with_rng(X, state, _rng(random_state))


@register_atom(witness_bernoulli_rbm_sample_visibles)
@icontract.require(lambda H, state, random_state: _hidden_matrix(H, state), "hidden samples must match fitted Bernoulli RBM state")
@icontract.ensure(lambda result, H, state: _binary_matrix_result_valid(result, (np.asarray(H).shape[0], state.n_features_in)), "visible samples must be binary")
def bernoulli_rbm_sample_visibles(
    H: NDArray[np.float64],
    state: BernoulliRBMState,
    *,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Draw visible Bernoulli states from hidden samples."""
    return _sample_visibles_with_rng(H, state, _rng(random_state))


@register_atom(witness_bernoulli_rbm_free_energy)
@icontract.require(lambda X, state: _matrix_against_state(X, state), "visible samples must match fitted Bernoulli RBM state")
@icontract.ensure(lambda result, X: _free_energy_result_valid(result, X), "free energy must be finite per sample")
def bernoulli_rbm_free_energy(X: NDArray[np.float64], state: BernoulliRBMState) -> NDArray[np.float64]:
    """Compute Bernoulli RBM free energy for each dense visible sample."""
    values = np.asarray(X, dtype=np.float64)
    hidden_terms = np.logaddexp(0.0, np.dot(values, state.components.T) + state.intercept_hidden)
    return -np.dot(values, state.intercept_visible) - hidden_terms.sum(axis=1)


@register_atom(witness_bernoulli_rbm_gibbs)
@icontract.require(lambda X, state, random_state: _matrix_against_state(X, state), "visible samples must match fitted Bernoulli RBM state")
@icontract.ensure(lambda result, X: _binary_matrix_result_valid(result, np.asarray(X).shape), "Gibbs output must be binary visible samples")
def bernoulli_rbm_gibbs(
    X: NDArray[np.float64],
    state: BernoulliRBMState,
    *,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Run one hidden-visible Gibbs transition from dense visible samples."""
    rng = _rng(random_state)
    h_sample = _sample_hiddens_with_rng(X, state, rng)
    return _sample_visibles_with_rng(h_sample, state, rng)


@register_atom(witness_bernoulli_rbm_partial_fit)
@icontract.require(lambda X, state, random_state: _matrix_against_state(X, state), "partial-fit samples must match fitted Bernoulli RBM state")
@icontract.ensure(lambda result, X, state: _fit_result_valid(result, X, state.n_components, state.learning_rate, state.batch_size, state.n_iter), "partial fit must preserve Bernoulli RBM state shape")
def bernoulli_rbm_partial_fit(
    X: NDArray[np.float64],
    state: BernoulliRBMState,
    *,
    random_state: int | None = None,
) -> BernoulliRBMState:
    """Apply one persistent contrastive-divergence update to fitted state."""
    return _fit_one_batch(np.asarray(X, dtype=np.float64), state, _rng(random_state))


@register_atom(witness_bernoulli_rbm_fit)
@icontract.require(lambda X, n_components, learning_rate, batch_size, n_iter, random_state: _fit_parameters_valid(X, n_components, learning_rate, batch_size, n_iter), "fit inputs must be finite dense Bernoulli visible samples")
@icontract.ensure(lambda result, X, n_components, learning_rate, batch_size, n_iter: _fit_result_valid(result, X, n_components, learning_rate, batch_size, n_iter), "fit must return a valid Bernoulli RBM state")
def bernoulli_rbm_fit(
    X: NDArray[np.float64],
    *,
    n_components: int = 256,
    learning_rate: float = 0.1,
    batch_size: int = 10,
    n_iter: int = 10,
    random_state: int | None = None,
) -> BernoulliRBMState:
    """Fit dense Bernoulli RBM parameters with persistent contrastive divergence."""
    values = np.asarray(X, dtype=np.float64)
    rng = _rng(random_state)
    state = BernoulliRBMState(
        components=np.asarray(rng.normal(0.0, 0.01, (n_components, values.shape[1])), dtype=np.float64, order="F"),
        intercept_hidden=np.zeros(n_components, dtype=np.float64),
        intercept_visible=np.zeros(values.shape[1], dtype=np.float64),
        h_samples=np.zeros((batch_size, n_components), dtype=np.float64),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        n_iter=int(n_iter),
        n_features_in=int(values.shape[1]),
        n_components=int(n_components),
    )
    n_batches = int(np.ceil(float(values.shape[0]) / batch_size))
    slices = [slice(start, min(start + batch_size, values.shape[0])) for start in range(0, n_batches * batch_size, batch_size)]
    for _ in range(n_iter):
        for batch_slice in slices:
            state = _fit_one_batch(values[batch_slice], state, rng)
    return state


@register_atom(witness_bernoulli_rbm_score_samples)
@icontract.require(lambda X, state, random_state: _matrix_against_state(X, state), "score samples must match fitted Bernoulli RBM state")
@icontract.ensure(lambda result, X: _free_energy_result_valid(result, X), "pseudo-likelihood must be finite per sample")
def bernoulli_rbm_score_samples(
    X: NDArray[np.float64],
    state: BernoulliRBMState,
    *,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Compute Bernoulli RBM pseudo-likelihood by flipping one visible feature."""
    values = np.asarray(X, dtype=np.float64)
    rng = _rng(random_state)
    indices = (np.arange(values.shape[0]), rng.randint(0, values.shape[1], values.shape[0]))
    corrupted = values.copy()
    corrupted[indices] = 1.0 - corrupted[indices]
    fe = bernoulli_rbm_free_energy(values, state)
    fe_corrupted = bernoulli_rbm_free_energy(corrupted, state)
    return -values.shape[1] * np.logaddexp(0.0, -(fe_corrupted - fe))
