"""Dense Gaussian mixture atoms adapted from scikit-learn."""

from __future__ import annotations

import math

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, digamma, gammaln, logsumexp

from sciona.ghost.registry import register_atom

from .state_models import BayesianGaussianMixtureDiagState, GaussianMixtureDiagState
from .witnesses import (
    witness_bayesian_gaussian_mixture_diag_fit,
    witness_bayesian_gaussian_mixture_diag_predict,
    witness_bayesian_gaussian_mixture_diag_predict_proba,
    witness_bayesian_gaussian_mixture_diag_score,
    witness_bayesian_gaussian_mixture_diag_score_samples,
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


def _bayesian_fit_options_valid(
    reg_covar: float,
    max_iter: int,
    tol: float,
    weight_concentration_prior_type: str,
    weight_concentration_prior: float | None,
    mean_precision_prior: float,
    degrees_of_freedom_prior: float | None,
    n_features: int,
) -> bool:
    return bool(
        _fit_options_valid(reg_covar, max_iter, tol)
        and max_iter >= 1
        and weight_concentration_prior_type in {"dirichlet_process", "dirichlet_distribution"}
        and (
            weight_concentration_prior is None
            or (
                isinstance(weight_concentration_prior, (int, float))
                and not isinstance(weight_concentration_prior, bool)
                and np.isfinite(float(weight_concentration_prior))
                and float(weight_concentration_prior) > 0.0
            )
        )
        and isinstance(mean_precision_prior, (int, float))
        and not isinstance(mean_precision_prior, bool)
        and np.isfinite(float(mean_precision_prior))
        and float(mean_precision_prior) > 0.0
        and (
            degrees_of_freedom_prior is None
            or (
                isinstance(degrees_of_freedom_prior, (int, float))
                and not isinstance(degrees_of_freedom_prior, bool)
                and np.isfinite(float(degrees_of_freedom_prior))
                and float(degrees_of_freedom_prior) > n_features - 1
            )
        )
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


def _bayesian_state_valid(state: BayesianGaussianMixtureDiagState) -> bool:
    component_count = state.weights.shape[0] if state.weights.ndim == 1 else -1
    feature_count = state.n_features_in
    weight_shape_ok = (
        (
            state.weight_concentration_prior_type == "dirichlet_process"
            and state.weight_concentration.shape == (2, component_count)
        )
        or (
            state.weight_concentration_prior_type == "dirichlet_distribution"
            and state.weight_concentration.shape == (component_count,)
        )
    )
    return bool(
        state.weight_concentration_prior_type in {"dirichlet_process", "dirichlet_distribution"}
        and state.weights.ndim == 1
        and state.means.shape == (component_count, feature_count)
        and state.covariances.shape == state.means.shape
        and state.precisions_cholesky.shape == state.means.shape
        and weight_shape_ok
        and state.mean_precision.shape == (component_count,)
        and state.mean_prior.shape == (feature_count,)
        and state.degrees_of_freedom.shape == (component_count,)
        and state.covariance_prior.shape == (feature_count,)
        and np.all(np.isfinite(state.weights))
        and np.all(state.weights >= 0.0)
        and np.allclose(np.sum(state.weights), 1.0)
        and np.all(np.isfinite(state.means))
        and np.all(np.isfinite(state.covariances))
        and np.all(state.covariances > 0.0)
        and np.all(np.isfinite(state.precisions_cholesky))
        and np.all(state.precisions_cholesky > 0.0)
        and np.all(np.isfinite(state.weight_concentration))
        and np.all(state.weight_concentration > 0.0)
        and np.all(np.isfinite(state.mean_precision))
        and np.all(state.mean_precision > 0.0)
        and np.all(np.isfinite(state.degrees_of_freedom))
        and np.all(state.degrees_of_freedom > feature_count - 1)
        and np.all(np.isfinite(state.covariance_prior))
        and np.all(state.covariance_prior > 0.0)
        and np.isfinite(state.lower_bound)
        and state.n_iter >= 1
    )


def _feature_count_matches(X: NDArray[np.float64], state: GaussianMixtureDiagState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in)


def _bayesian_feature_count_matches(X: NDArray[np.float64], state: BayesianGaussianMixtureDiagState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in)


def _responsibilities_valid(X: NDArray[np.float64], responsibilities: NDArray[np.float64]) -> bool:
    try:
        values_x = np.asarray(X, dtype=np.float64)
        resp = np.asarray(responsibilities, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values_x.ndim == 2
        and resp.ndim == 2
        and resp.shape[0] == values_x.shape[0]
        and 1 <= resp.shape[1] <= values_x.shape[0]
        and np.all(np.isfinite(resp))
        and np.all(resp >= 0.0)
        and np.all(np.sum(resp, axis=0) > 0.0)
        and np.allclose(np.sum(resp, axis=1), 1.0)
    )


def _optional_prior_shapes_valid(
    X: NDArray[np.float64],
    mean_prior: NDArray[np.float64] | None,
    covariance_prior: NDArray[np.float64] | None,
) -> bool:
    n_features = np.asarray(X).shape[1]
    mean_ok = mean_prior is None or np.asarray(mean_prior).shape == (n_features,)
    covariance_ok = covariance_prior is None or np.asarray(covariance_prior).shape == (n_features,)
    return bool(mean_ok and covariance_ok)


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


def _log_dirichlet_norm(dirichlet_concentration: NDArray[np.float64]) -> float:
    return float(gammaln(np.sum(dirichlet_concentration)) - np.sum(gammaln(dirichlet_concentration)))


def _log_wishart_norm(
    degrees_of_freedom: NDArray[np.float64],
    log_det_precisions_chol: NDArray[np.float64],
    n_features: int,
) -> NDArray[np.float64]:
    return -(
        degrees_of_freedom * log_det_precisions_chol
        + degrees_of_freedom * n_features * 0.5 * math.log(2.0)
        + np.sum(gammaln(0.5 * (degrees_of_freedom - np.arange(n_features)[:, np.newaxis])), axis=0)
    )


def _bayesian_weights(
    weight_concentration: NDArray[np.float64],
    weight_concentration_prior_type: str,
) -> NDArray[np.float64]:
    if weight_concentration_prior_type == "dirichlet_process":
        weight_sum = weight_concentration[0] + weight_concentration[1]
        tmp = weight_concentration[1] / weight_sum
        weights = weight_concentration[0] / weight_sum * np.hstack((1.0, np.cumprod(tmp[:-1])))
        return weights / np.sum(weights)
    return weight_concentration / np.sum(weight_concentration)


def _bayesian_log_weights(state: BayesianGaussianMixtureDiagState) -> NDArray[np.float64]:
    if state.weight_concentration_prior_type == "dirichlet_process":
        digamma_sum = digamma(state.weight_concentration[0] + state.weight_concentration[1])
        digamma_a = digamma(state.weight_concentration[0])
        digamma_b = digamma(state.weight_concentration[1])
        return digamma_a - digamma_sum + np.hstack((0.0, np.cumsum(digamma_b - digamma_sum)[:-1]))
    return digamma(state.weight_concentration) - digamma(np.sum(state.weight_concentration))


def _bayesian_m_step_diag(
    X: NDArray[np.float64],
    log_resp: NDArray[np.float64],
    reg_covar: float,
    weight_concentration_prior_type: str,
    weight_concentration_prior: float,
    mean_precision_prior: float,
    mean_prior: NDArray[np.float64],
    degrees_of_freedom_prior: float,
    covariance_prior: NDArray[np.float64],
    lower_bound: float,
    lower_bounds: list[float],
    n_iter: int,
    converged: bool,
) -> BayesianGaussianMixtureDiagState:
    resp = np.exp(log_resp)
    nk = np.sum(resp, axis=0) + 10.0 * np.finfo(resp.dtype).eps
    xk = (resp.T @ X) / nk[:, np.newaxis]
    sk = (resp.T @ (X * X)) / nk[:, np.newaxis] - xk**2 + reg_covar
    if weight_concentration_prior_type == "dirichlet_process":
        weight_concentration = np.vstack(
            (
                1.0 + nk,
                weight_concentration_prior + np.hstack((np.cumsum(nk[::-1])[-2::-1], 0.0)),
            )
        )
    else:
        weight_concentration = weight_concentration_prior + nk
    mean_precision = mean_precision_prior + nk
    means = (mean_precision_prior * mean_prior + nk[:, np.newaxis] * xk) / mean_precision[:, np.newaxis]
    degrees_of_freedom = degrees_of_freedom_prior + nk
    diff = xk - mean_prior
    covariances = covariance_prior + nk[:, np.newaxis] * (
        sk + (mean_precision_prior / mean_precision)[:, np.newaxis] * np.square(diff)
    )
    covariances = covariances / degrees_of_freedom[:, np.newaxis]
    precisions_cholesky = 1.0 / np.sqrt(covariances)
    weights = _bayesian_weights(weight_concentration, weight_concentration_prior_type)
    return BayesianGaussianMixtureDiagState(
        weights=weights,
        means=means,
        covariances=covariances,
        precisions_cholesky=precisions_cholesky,
        weight_concentration=weight_concentration,
        weight_concentration_prior=weight_concentration_prior,
        weight_concentration_prior_type=weight_concentration_prior_type,
        mean_precision=mean_precision,
        mean_precision_prior=mean_precision_prior,
        mean_prior=mean_prior,
        degrees_of_freedom=degrees_of_freedom,
        degrees_of_freedom_prior=degrees_of_freedom_prior,
        covariance_prior=covariance_prior,
        converged=converged,
        n_iter=n_iter,
        lower_bound=lower_bound,
        lower_bounds=np.asarray(lower_bounds, dtype=np.float64),
        reg_covar=reg_covar,
        n_features_in=X.shape[1],
    )


def _bayesian_estimate_log_prob(
    X: NDArray[np.float64],
    state: BayesianGaussianMixtureDiagState,
) -> NDArray[np.float64]:
    _, n_features = X.shape
    log_gauss = _estimate_log_gaussian_prob_diag(X, state.means, state.precisions_cholesky) - 0.5 * n_features * np.log(
        state.degrees_of_freedom
    )
    log_lambda = n_features * math.log(2.0) + np.sum(
        digamma(0.5 * (state.degrees_of_freedom - np.arange(0, n_features)[:, np.newaxis])),
        axis=0,
    )
    return log_gauss + 0.5 * (log_lambda - n_features / state.mean_precision)


def _bayesian_estimate_log_prob_resp(
    X: NDArray[np.float64],
    state: BayesianGaussianMixtureDiagState,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    weighted_log_prob = _bayesian_estimate_log_prob(X, state) + _bayesian_log_weights(state)
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    return np.asarray(log_prob_norm, dtype=np.float64), weighted_log_prob - log_prob_norm[:, np.newaxis]


def _bayesian_lower_bound(
    state: BayesianGaussianMixtureDiagState,
    log_resp: NDArray[np.float64],
) -> float:
    n_features = state.mean_prior.shape[0]
    log_det_precisions_chol = np.sum(np.log(state.precisions_cholesky), axis=1) - 0.5 * n_features * np.log(
        state.degrees_of_freedom
    )
    log_wishart = np.sum(_log_wishart_norm(state.degrees_of_freedom, log_det_precisions_chol, n_features))
    if state.weight_concentration_prior_type == "dirichlet_process":
        log_norm_weight = -np.sum(betaln(state.weight_concentration[0], state.weight_concentration[1]))
    else:
        log_norm_weight = _log_dirichlet_norm(state.weight_concentration)
    return float(
        -np.sum(np.exp(log_resp) * log_resp)
        - log_wishart
        - log_norm_weight
        - 0.5 * n_features * np.sum(np.log(state.mean_precision))
    )


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


@register_atom(witness_bayesian_gaussian_mixture_diag_fit)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda X, initial_responsibilities: _responsibilities_valid(X, initial_responsibilities))
@icontract.require(lambda X, mean_prior, covariance_prior: _optional_prior_shapes_valid(X, mean_prior, covariance_prior))
@icontract.require(lambda X, reg_covar, max_iter, tol, weight_concentration_prior_type, weight_concentration_prior, mean_precision_prior, degrees_of_freedom_prior: _bayesian_fit_options_valid(reg_covar, max_iter, tol, weight_concentration_prior_type, weight_concentration_prior, mean_precision_prior, degrees_of_freedom_prior, np.asarray(X).shape[1]))
@icontract.ensure(lambda result: _bayesian_state_valid(result))
def bayesian_gaussian_mixture_diag_fit(
    X: NDArray[np.float64],
    initial_responsibilities: NDArray[np.float64],
    *,
    reg_covar: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-3,
    weight_concentration_prior_type: str = "dirichlet_process",
    weight_concentration_prior: float | None = None,
    mean_precision_prior: float = 1.0,
    mean_prior: NDArray[np.float64] | None = None,
    degrees_of_freedom_prior: float | None = None,
    covariance_prior: NDArray[np.float64] | None = None,
) -> BayesianGaussianMixtureDiagState:
    """Fit a dense diagonal variational Gaussian mixture from responsibilities."""
    values = np.asarray(X, dtype=np.float64)
    resp = np.asarray(initial_responsibilities, dtype=np.float64)
    n_components = resp.shape[1]
    n_features = values.shape[1]
    resolved_weight_prior = float(weight_concentration_prior) if weight_concentration_prior is not None else 1.0 / n_components
    resolved_mean_precision_prior = float(mean_precision_prior)
    resolved_mean_prior = (
        np.asarray(mean_prior, dtype=np.float64).copy() if mean_prior is not None else np.mean(values, axis=0)
    )
    resolved_degrees_prior = float(degrees_of_freedom_prior) if degrees_of_freedom_prior is not None else float(n_features)
    resolved_covariance_prior = (
        np.asarray(covariance_prior, dtype=np.float64).copy()
        if covariance_prior is not None
        else np.var(values, axis=0, ddof=1)
    )
    initial_log_resp = np.full_like(resp, -np.inf, dtype=np.float64)
    np.log(resp, out=initial_log_resp, where=resp > 0.0)
    state = _bayesian_m_step_diag(
        values,
        initial_log_resp,
        float(reg_covar),
        weight_concentration_prior_type,
        resolved_weight_prior,
        resolved_mean_precision_prior,
        resolved_mean_prior,
        resolved_degrees_prior,
        resolved_covariance_prior,
        lower_bound=0.0,
        lower_bounds=[],
        n_iter=1,
        converged=False,
    )
    lower_bound = -np.inf
    lower_bounds: list[float] = []
    converged = False
    for n_iter in range(1, max_iter + 1):
        previous_lower_bound = lower_bound
        log_prob_norm, log_resp = _bayesian_estimate_log_prob_resp(values, state)
        state = _bayesian_m_step_diag(
            values,
            log_resp,
            float(reg_covar),
            weight_concentration_prior_type,
            resolved_weight_prior,
            resolved_mean_precision_prior,
            resolved_mean_prior,
            resolved_degrees_prior,
            resolved_covariance_prior,
            lower_bound=state.lower_bound,
            lower_bounds=lower_bounds,
            n_iter=n_iter,
            converged=False,
        )
        lower_bound = _bayesian_lower_bound(state, log_resp)
        lower_bounds.append(lower_bound)
        state = BayesianGaussianMixtureDiagState(
            weights=state.weights,
            means=state.means,
            covariances=state.covariances,
            precisions_cholesky=state.precisions_cholesky,
            weight_concentration=state.weight_concentration,
            weight_concentration_prior=state.weight_concentration_prior,
            weight_concentration_prior_type=state.weight_concentration_prior_type,
            mean_precision=state.mean_precision,
            mean_precision_prior=state.mean_precision_prior,
            mean_prior=state.mean_prior,
            degrees_of_freedom=state.degrees_of_freedom,
            degrees_of_freedom_prior=state.degrees_of_freedom_prior,
            covariance_prior=state.covariance_prior,
            converged=False,
            n_iter=n_iter,
            lower_bound=lower_bound,
            lower_bounds=np.asarray(lower_bounds, dtype=np.float64),
            reg_covar=state.reg_covar,
            n_features_in=state.n_features_in,
        )
        if abs(lower_bound - previous_lower_bound) < tol:
            converged = True
            break
        _ = log_prob_norm

    return BayesianGaussianMixtureDiagState(
        weights=state.weights,
        means=state.means,
        covariances=state.covariances,
        precisions_cholesky=state.precisions_cholesky,
        weight_concentration=state.weight_concentration,
        weight_concentration_prior=state.weight_concentration_prior,
        weight_concentration_prior_type=state.weight_concentration_prior_type,
        mean_precision=state.mean_precision,
        mean_precision_prior=state.mean_precision_prior,
        mean_prior=state.mean_prior,
        degrees_of_freedom=state.degrees_of_freedom,
        degrees_of_freedom_prior=state.degrees_of_freedom_prior,
        covariance_prior=state.covariance_prior,
        converged=converged,
        n_iter=state.n_iter,
        lower_bound=state.lower_bound,
        lower_bounds=state.lower_bounds,
        reg_covar=state.reg_covar,
        n_features_in=state.n_features_in,
    )


@register_atom(witness_bayesian_gaussian_mixture_diag_score_samples)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda state: _bayesian_state_valid(state))
@icontract.require(lambda X, state: _bayesian_feature_count_matches(X, state))
@icontract.ensure(lambda X, result: _score_vector_valid(result, np.asarray(X).shape[0]))
def bayesian_gaussian_mixture_diag_score_samples(
    X: NDArray[np.float64],
    state: BayesianGaussianMixtureDiagState,
) -> NDArray[np.float64]:
    """Compute per-sample variational mixture log scores."""
    values = np.asarray(X, dtype=np.float64)
    weighted_log_prob = _bayesian_estimate_log_prob(values, state) + _bayesian_log_weights(state)
    return np.asarray(logsumexp(weighted_log_prob, axis=1), dtype=np.float64)


@register_atom(witness_bayesian_gaussian_mixture_diag_score)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda state: _bayesian_state_valid(state))
@icontract.require(lambda X, state: _bayesian_feature_count_matches(X, state))
@icontract.ensure(lambda result: np.isfinite(result))
def bayesian_gaussian_mixture_diag_score(
    X: NDArray[np.float64],
    state: BayesianGaussianMixtureDiagState,
) -> float:
    """Compute the mean variational mixture log score."""
    return float(np.mean(bayesian_gaussian_mixture_diag_score_samples(X, state)))


@register_atom(witness_bayesian_gaussian_mixture_diag_predict_proba)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda state: _bayesian_state_valid(state))
@icontract.require(lambda X, state: _bayesian_feature_count_matches(X, state))
@icontract.ensure(lambda X, state, result: _probability_rows_valid(result, np.asarray(X).shape[0], state.weights.shape[0]))
def bayesian_gaussian_mixture_diag_predict_proba(
    X: NDArray[np.float64],
    state: BayesianGaussianMixtureDiagState,
) -> NDArray[np.float64]:
    """Compute posterior component probabilities for a variational mixture."""
    values = np.asarray(X, dtype=np.float64)
    _, log_resp = _bayesian_estimate_log_prob_resp(values, state)
    return np.exp(log_resp)


@register_atom(witness_bayesian_gaussian_mixture_diag_predict)
@icontract.require(lambda X: _matrix_2d(X))
@icontract.require(lambda X: _finite_matrix(X))
@icontract.require(lambda state: _bayesian_state_valid(state))
@icontract.require(lambda X, state: _bayesian_feature_count_matches(X, state))
@icontract.ensure(lambda X, state, result: _label_vector_valid(result, np.asarray(X).shape[0], state.weights.shape[0]))
def bayesian_gaussian_mixture_diag_predict(
    X: NDArray[np.float64],
    state: BayesianGaussianMixtureDiagState,
) -> NDArray[np.int64]:
    """Assign samples to their highest-posterior variational component."""
    return np.asarray(np.argmax(bayesian_gaussian_mixture_diag_predict_proba(X, state), axis=1), dtype=np.int64)
