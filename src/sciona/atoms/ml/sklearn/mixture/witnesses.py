"""Ghost witnesses for sklearn mixture atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import GaussianMixtureDiagState


def _check_matrix(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return int(X.shape[0]), int(X.shape[1])


def _check_diag_state(X: AbstractArray, state: GaussianMixtureDiagState) -> tuple[int, int]:
    n_samples, n_features = _check_matrix(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.weights.shape[0] != state.means.shape[0]:
        raise ValueError("state component counts must agree")
    return n_samples, state.weights.shape[0]


def witness_gaussian_mixture_diag_fit(
    X: AbstractArray,
    initial_weights: AbstractArray,
    initial_means: AbstractArray,
    initial_variances: AbstractArray,
    *,
    reg_covar: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> AbstractArray:
    """Describe fitting a diagonal Gaussian mixture from explicit initialization."""
    n_samples, n_features = _check_matrix(X)
    if len(initial_weights.shape) != 1:
        raise ValueError("initial_weights must be 1D")
    n_components = int(initial_weights.shape[0])
    if initial_means.shape != (n_components, n_features):
        raise ValueError("initial_means shape must match components and features")
    if initial_variances.shape != (n_components, n_features):
        raise ValueError("initial_variances shape must match components and features")
    if n_samples < n_components:
        raise ValueError("sample count must cover all mixture components")
    if reg_covar < 0 or max_iter < 0 or tol < 0:
        raise ValueError("regularization and iteration parameters must be nonnegative")
    return AbstractArray(shape=(n_components, n_features), dtype="float64")


def witness_gaussian_mixture_diag_score_samples(X: AbstractArray, state: GaussianMixtureDiagState) -> AbstractArray:
    """Describe per-sample mixture log-likelihood scores."""
    n_samples, _ = _check_diag_state(X, state)
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_gaussian_mixture_diag_score(X: AbstractArray, state: GaussianMixtureDiagState) -> AbstractArray:
    """Describe the mean mixture log-likelihood score."""
    _check_diag_state(X, state)
    return AbstractArray(shape=(), dtype="float64")


def witness_gaussian_mixture_diag_predict_proba(X: AbstractArray, state: GaussianMixtureDiagState) -> AbstractArray:
    """Describe posterior component responsibilities."""
    n_samples, n_components = _check_diag_state(X, state)
    return AbstractArray(shape=(n_samples, n_components), dtype="float64")


def witness_gaussian_mixture_diag_predict(X: AbstractArray, state: GaussianMixtureDiagState) -> AbstractArray:
    """Describe maximum-posterior component labels."""
    n_samples, _ = _check_diag_state(X, state)
    return AbstractArray(shape=(n_samples,), dtype="int64")


def witness_gaussian_mixture_diag_bic(X: AbstractArray, state: GaussianMixtureDiagState) -> AbstractArray:
    """Describe Bayesian information criterion calculation."""
    _check_diag_state(X, state)
    return AbstractArray(shape=(), dtype="float64")


def witness_gaussian_mixture_diag_aic(X: AbstractArray, state: GaussianMixtureDiagState) -> AbstractArray:
    """Describe Akaike information criterion calculation."""
    _check_diag_state(X, state)
    return AbstractArray(shape=(), dtype="float64")
