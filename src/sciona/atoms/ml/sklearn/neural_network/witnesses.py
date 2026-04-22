"""Ghost witnesses for sklearn neural-network atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import BernoulliRBMState


def witness_bernoulli_rbm_mean_hiddens(X: AbstractArray, state: BernoulliRBMState) -> AbstractArray:
    """Describe hidden activation probabilities for dense visible samples."""
    if len(X.shape) != 2 or X.shape[1] != state.n_features_in:
        raise ValueError("X must be 2D with the fitted visible feature count")
    return AbstractArray(shape=(int(X.shape[0]), state.n_components), dtype="float64")


def witness_bernoulli_rbm_sample_hiddens(
    X: AbstractArray,
    state: BernoulliRBMState,
    *,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe Bernoulli hidden samples drawn from activation probabilities."""
    _ = random_state
    return witness_bernoulli_rbm_mean_hiddens(X, state)


def witness_bernoulli_rbm_sample_visibles(
    H: AbstractArray,
    state: BernoulliRBMState,
    *,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe visible Bernoulli samples drawn from hidden states."""
    _ = random_state
    if len(H.shape) != 2 or H.shape[1] != state.n_components:
        raise ValueError("H must be 2D with the fitted hidden component count")
    return AbstractArray(shape=(int(H.shape[0]), state.n_features_in), dtype="float64")


def witness_bernoulli_rbm_free_energy(X: AbstractArray, state: BernoulliRBMState) -> AbstractArray:
    """Describe one free-energy value for each dense visible sample."""
    if len(X.shape) != 2 or X.shape[1] != state.n_features_in:
        raise ValueError("X must be 2D with the fitted visible feature count")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_bernoulli_rbm_gibbs(
    X: AbstractArray,
    state: BernoulliRBMState,
    *,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe one visible-to-hidden-to-visible Gibbs transition."""
    _ = random_state
    if len(X.shape) != 2 or X.shape[1] != state.n_features_in:
        raise ValueError("X must be 2D with the fitted visible feature count")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_bernoulli_rbm_partial_fit(
    X: AbstractArray,
    state: BernoulliRBMState,
    *,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe a persistent contrastive-divergence update of fitted state."""
    _ = random_state
    if len(X.shape) != 2 or X.shape[1] != state.n_features_in:
        raise ValueError("X must be 2D with the fitted visible feature count")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_bernoulli_rbm_fit(
    X: AbstractArray,
    *,
    n_components: int = 256,
    learning_rate: float = 0.1,
    batch_size: int = 10,
    n_iter: int = 10,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe learned Bernoulli restricted Boltzmann machine state."""
    _ = random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if n_components < 1 or learning_rate <= 0.0 or batch_size < 1 or n_iter < 0:
        raise ValueError("invalid Bernoulli RBM fit parameters")
    return AbstractArray(shape=(n_components, int(X.shape[1])), dtype="float64")


def witness_bernoulli_rbm_score_samples(
    X: AbstractArray,
    state: BernoulliRBMState,
    *,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe pseudo-likelihood scores for dense visible samples."""
    _ = random_state
    return witness_bernoulli_rbm_free_energy(X, state)
