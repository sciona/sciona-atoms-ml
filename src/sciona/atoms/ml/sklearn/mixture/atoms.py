"""Dense Gaussian mixture atoms adapted from scikit-learn."""

from __future__ import annotations

import math

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from sciona.ghost.registry import register_atom

from .state_models import GaussianMixtureDiagState
from .witnesses import (
    witness_gaussian_mixture_diag_aic,
    witness_gaussian_mixture_diag_bic,
    witness_gaussian_mixture_diag_fit,
    witness_gaussian_mixture_diag_predict,
    witness_gaussian_mixture_diag_predict_proba,
    witness_gaussian_mixture_diag_score,
    witness_gaussian_mixture_diag_score_samples,
)


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2)


def _finite_matrix(X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and np.all(np.isfinite(values)))


def _weights_valid(weights: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 1
        and values.shape[0] >= 1
        and np.all(np.isfinite(values))
        and np.all(values > 0.0)
        and np.allclose(np.sum(values), 1.0)
    )


def _initial_shapes_valid(
    X: NDArray[np.float64],
    initial_weights: NDArray[np.float64],
    initial_means: NDArray[np.float64],
    initial_variances: NDArray[np.float64],
) -> bool:
    values_x = np.asarray(X)
    weights = np.asarray(initial_weights)
    means = np.asarray(initial_means)
    variances = np.asarray(initial_variances)
    return bool(
        values_x.ndim == 2
        and weights.ndim == 1
        and means.shape == (weights.shape[0], values_x.shape[1])
        and variances.shape == means.shape
        and values_x.shape[0] >= weights.shape[0]
    )


def _positive_finite_matrix(X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and np.all(np.isfinite(values)) and np.all(values > 0.0))


def _fit_options_valid(reg_covar: float, max_iter: int, tol: float) -> bool:
    return bool(
        isinstance(reg_covar, (int, float))
        and not isinstance(reg_covar, bool)
        and np.isfinite(float(reg_covar))
        and float(reg_covar) >= 0.0
        and isinstance(max_iter, int)
        and not isinstance(max_iter, bool)
        and max_iter >= 0
        and isinstance(tol, (int, float))
        and not isinstance(tol, bool)
        and np.isfinite(float(tol))
        and float(tol) >= 0.0
    )


def _state_valid(state: GaussianMixtureDiagState) -> bool:
    return bool(
        state.weights.ndim == 1
        and state.means.ndim == 2
        and state.covariances.shape == state.means.shape
        and state.precisions_cholesky.shape == state.means.shape
        and state.weights.shape[0] == state.means.shape[0]
        and state.means.shape[1] == state.n_features_in
        and np.all(np.isfinite(state.weights))
        and np.all(state.weights > 0.0)
        and np.allclose(np.sum(state.weights), 1.0)
        and np.all(np.isfinite(state.means))
        and np.all(np.isfinite(state.covariances))
        and np.all(state.covariances > 0.0)
        and np.all(np.isfinite(state.precisions_cholesky))
        and np.all(state.precisions_cholesky > 0.0)
        and np.isfinite(state.lower_bound)
        and state.n_iter >= 0
    )


def _feature_count_matches(X: NDArray[np.float64], state: GaussianMixtureDiagState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in)


def _probability_rows_valid(result: NDArray[np.float64], n_samples: int, n_components: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (n_samples, n_components)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _score_vector_valid(result: NDArray[np.float64], n_samples: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (n_samples,) and np.all(np.isfinite(values)))


def _label_vector_valid(result: NDArray[np.int64], n_samples: int, n_components: int) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (n_samples,) and np.all(values >= 0) and np.all(values < n_components))


def _estimate_log_gaussian_prob_diag(
    X: NDArray[np.float64],
    means: NDArray[np.float64],
    precisions_cholesky: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    precisions = precisions_cholesky**2
    log_det = np.sum(np.log(precisions_cholesky), axis=1)
    log_prob = (
        np.sum((means**2 * precisions), axis=1)
        - 2.0 * (X @ (means * precisions).T)
        + (X**2 @ precisions.T)
    )
    return -0.5 * (n_features * math.log(2.0 * math.pi) + log_prob) + log_det.reshape(
        1, n_components
    )


def _estimate_log_prob_resp(
    X: NDArray[np.float64],
    state: GaussianMixtureDiagState,
) -> tuple[float, NDArray[np.float64]]:
    weighted_log_prob = _estimate_log_gaussian_prob_diag(X, state.means, state.precisions_cholesky) + np.log(
        state.weights
    )
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    return float(np.mean(log_prob_norm)), weighted_log_prob - log_prob_norm[:, np.newaxis]


def _m_step_diag(
    X: NDArray[np.float64],
    log_resp: NDArray[np.float64],
    reg_covar: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    resp = np.exp(log_resp)
    nk = np.sum(resp, axis=0) + 10.0 * np.finfo(resp.dtype).eps
    weights = nk / np.sum(nk)
    means = (resp.T @ X) / nk[:, np.newaxis]
    covariances = (resp.T @ (X * X)) / nk[:, np.newaxis] - means**2 + reg_covar
    precisions_cholesky = 1.0 / np.sqrt(covariances)
    return weights, means, covariances, precisions_cholesky


def _n_parameters_diag(state: GaussianMixtureDiagState) -> int:
    n_components, n_features = state.means.shape
    return int(2 * n_components * n_features + n_components - 1)


@register_atom(witness_gaussian_mixture_diag_fit)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda initial_weights: _weights_valid(initial_weights))
@icontract.require(lambda initial_variances: _positive_finite_matrix(initial_variances))
@icontract.require(lambda X, initial_weights, initial_means, initial_variances: _initial_shapes_valid(X, initial_weights, initial_means, initial_variances))
@icontract.require(lambda reg_covar, max_iter, tol: _fit_options_valid(reg_covar, max_iter, tol))
@icontract.ensure(lambda result: _state_valid(result))
def gaussian_mixture_diag_fit(
    X: NDArray[np.float64],
    initial_weights: NDArray[np.float64],
    initial_means: NDArray[np.float64],
    initial_variances: NDArray[np.float64],
    *,
    reg_covar: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> GaussianMixtureDiagState:
    """Fit a dense diagonal-covariance Gaussian mixture from explicit parameters."""
    values = np.asarray(X, dtype=np.float64)
    weights = np.asarray(initial_weights, dtype=np.float64).copy()
    means = np.asarray(initial_means, dtype=np.float64).copy()
    covariances = np.asarray(initial_variances, dtype=np.float64).copy()
    precisions_cholesky = 1.0 / np.sqrt(covariances)
    state = GaussianMixtureDiagState(
        weights=weights,
        means=means,
        covariances=covariances,
        precisions_cholesky=precisions_cholesky,
        converged=False,
        n_iter=0,
        lower_bound=-np.inf,
        lower_bounds=np.empty(0, dtype=np.float64),
        reg_covar=float(reg_covar),
        n_features_in=values.shape[1],
    )
    if max_iter == 0:
        lower_bound, _ = _estimate_log_prob_resp(values, state)
        return GaussianMixtureDiagState(
            weights=state.weights,
            means=state.means,
            covariances=state.covariances,
            precisions_cholesky=state.precisions_cholesky,
            converged=False,
            n_iter=0,
            lower_bound=lower_bound,
            lower_bounds=np.empty(0, dtype=np.float64),
            reg_covar=float(reg_covar),
            n_features_in=values.shape[1],
        )

    lower_bound = -np.inf
    lower_bounds: list[float] = []
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        previous_lower_bound = lower_bound
        lower_bound, log_resp = _estimate_log_prob_resp(values, state)
        weights, means, covariances, precisions_cholesky = _m_step_diag(values, log_resp, float(reg_covar))
        state = GaussianMixtureDiagState(
            weights=weights,
            means=means,
            covariances=covariances,
            precisions_cholesky=precisions_cholesky,
            converged=False,
            n_iter=n_iter,
            lower_bound=lower_bound,
            lower_bounds=np.asarray(lower_bounds + [lower_bound], dtype=np.float64),
            reg_covar=float(reg_covar),
            n_features_in=values.shape[1],
        )
        lower_bounds.append(lower_bound)
        if abs(lower_bound - previous_lower_bound) < tol:
            converged = True
            break

    return GaussianMixtureDiagState(
        weights=state.weights,
        means=state.means,
        covariances=state.covariances,
        precisions_cholesky=state.precisions_cholesky,
        converged=converged,
        n_iter=n_iter,
        lower_bound=lower_bound,
        lower_bounds=np.asarray(lower_bounds, dtype=np.float64),
        reg_covar=float(reg_covar),
        n_features_in=values.shape[1],
    )


@register_atom(witness_gaussian_mixture_diag_score_samples)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda state: _state_valid(state))
@icontract.require(lambda X, state: _feature_count_matches(X, state))
@icontract.ensure(lambda X, result: _score_vector_valid(result, np.asarray(X).shape[0]))
def gaussian_mixture_diag_score_samples(
    X: NDArray[np.float64],
    state: GaussianMixtureDiagState,
) -> NDArray[np.float64]:
    """Compute per-sample log likelihood under a diagonal Gaussian mixture."""
    values = np.asarray(X, dtype=np.float64)
    weighted_log_prob = _estimate_log_gaussian_prob_diag(values, state.means, state.precisions_cholesky) + np.log(
        state.weights
    )
    return np.asarray(logsumexp(weighted_log_prob, axis=1), dtype=np.float64)


@register_atom(witness_gaussian_mixture_diag_score)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda state: _state_valid(state))
@icontract.require(lambda X, state: _feature_count_matches(X, state))
@icontract.ensure(lambda result: np.isfinite(result))
def gaussian_mixture_diag_score(
    X: NDArray[np.float64],
    state: GaussianMixtureDiagState,
) -> float:
    """Compute the mean log likelihood under a diagonal Gaussian mixture."""
    return float(np.mean(gaussian_mixture_diag_score_samples(X, state)))


@register_atom(witness_gaussian_mixture_diag_predict_proba)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda state: _state_valid(state))
@icontract.require(lambda X, state: _feature_count_matches(X, state))
@icontract.ensure(lambda X, state, result: _probability_rows_valid(result, np.asarray(X).shape[0], state.weights.shape[0]))
def gaussian_mixture_diag_predict_proba(
    X: NDArray[np.float64],
    state: GaussianMixtureDiagState,
) -> NDArray[np.float64]:
    """Compute posterior component probabilities for each sample."""
    values = np.asarray(X, dtype=np.float64)
    weighted_log_prob = _estimate_log_gaussian_prob_diag(values, state.means, state.precisions_cholesky) + np.log(
        state.weights
    )
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    return np.exp(weighted_log_prob - log_prob_norm[:, np.newaxis])


@register_atom(witness_gaussian_mixture_diag_predict)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda state: _state_valid(state))
@icontract.require(lambda X, state: _feature_count_matches(X, state))
@icontract.ensure(lambda X, state, result: _label_vector_valid(result, np.asarray(X).shape[0], state.weights.shape[0]))
def gaussian_mixture_diag_predict(
    X: NDArray[np.float64],
    state: GaussianMixtureDiagState,
) -> NDArray[np.int64]:
    """Assign each sample to the most likely Gaussian component."""
    return np.asarray(np.argmax(gaussian_mixture_diag_predict_proba(X, state), axis=1), dtype=np.int64)


@register_atom(witness_gaussian_mixture_diag_bic)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda state: _state_valid(state))
@icontract.require(lambda X, state: _feature_count_matches(X, state))
@icontract.ensure(lambda result: np.isfinite(result))
def gaussian_mixture_diag_bic(
    X: NDArray[np.float64],
    state: GaussianMixtureDiagState,
) -> float:
    """Compute the Bayesian information criterion for a fitted diagonal mixture."""
    n_samples = np.asarray(X).shape[0]
    score_sum = float(np.sum(gaussian_mixture_diag_score_samples(X, state)))
    return float(-2.0 * score_sum + _n_parameters_diag(state) * math.log(n_samples))


@register_atom(witness_gaussian_mixture_diag_aic)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda state: _state_valid(state))
@icontract.require(lambda X, state: _feature_count_matches(X, state))
@icontract.ensure(lambda result: np.isfinite(result))
def gaussian_mixture_diag_aic(
    X: NDArray[np.float64],
    state: GaussianMixtureDiagState,
) -> float:
    """Compute the Akaike information criterion for a fitted diagonal mixture."""
    score_sum = float(np.sum(gaussian_mixture_diag_score_samples(X, state)))
    return float(-2.0 * score_sum + 2.0 * _n_parameters_diag(state))
