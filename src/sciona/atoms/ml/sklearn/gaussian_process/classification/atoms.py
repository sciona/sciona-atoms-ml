"""Gaussian-process classification Laplace atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_solve, cholesky, solve
from scipy.special import erf, expit

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_classifier_laplace_log_marginal_likelihood,
    witness_gp_classifier_laplace_newton_step,
    witness_gp_classifier_posterior_cross_solve,
    witness_gp_classifier_posterior_mean,
    witness_gp_classifier_posterior_variance,
    witness_gp_classifier_predictive_proba,
    witness_gp_classifier_predictive_probability,
)

LaplaceNewtonStep = tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]

_COEFS = np.array(
    [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654],
    dtype=np.float64,
)
_LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39], dtype=np.float64)


def _finite_vector(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_square_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] == array.shape[1]
        and array.shape[0] >= 1
        and np.all(np.isfinite(array))
    )


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _lower_cholesky_factor(values: NDArray[np.float64]) -> bool:
    if not _finite_square_matrix(values):
        return False
    factor = np.asarray(values, dtype=np.float64)
    return bool(
        np.allclose(factor, np.tril(factor))
        and np.all(np.diag(factor) > 0.0)
    )


def _binary_targets(values: NDArray[np.float64]) -> bool:
    if not _finite_vector(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isin(array, np.array([0.0, 1.0], dtype=np.float64))))


def _symmetric_positive_semidefinite(values: NDArray[np.float64]) -> bool:
    if not _finite_square_matrix(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    if not np.allclose(array, array.T):
        return False
    eigenvalues = np.linalg.eigvalsh(array)
    return bool(np.all(eigenvalues >= -1e-10))


def _laplace_inputs_valid(K: NDArray[np.float64], y_train: NDArray[np.float64], f: NDArray[np.float64]) -> bool:
    if not (_symmetric_positive_semidefinite(K) and _binary_targets(y_train) and _finite_vector(f)):
        return False
    n_samples = np.asarray(K, dtype=np.float64).shape[0]
    return bool(np.asarray(y_train, dtype=np.float64).shape == (n_samples,) and np.asarray(f, dtype=np.float64).shape == (n_samples,))


def _laplace_step_result_valid(result: LaplaceNewtonStep, K: NDArray[np.float64]) -> bool:
    if not isinstance(result, tuple) or len(result) != 6:
        return False
    f_next, pi, w_sr, L, b, a = result
    n_samples = np.asarray(K, dtype=np.float64).shape[0]
    vectors = (f_next, pi, w_sr, b, a)
    return bool(
        all(_finite_vector(vector) and np.asarray(vector).shape == (n_samples,) for vector in vectors)
        and _lower_cholesky_factor(L)
        and np.asarray(L).shape == (n_samples, n_samples)
        and np.all(np.asarray(pi) > 0.0)
        and np.all(np.asarray(pi) < 1.0)
        and np.all(np.asarray(w_sr) >= 0.0)
    )


def _laplace_lml_inputs_valid(y_train: NDArray[np.float64], f: NDArray[np.float64], a: NDArray[np.float64], L: NDArray[np.float64]) -> bool:
    if not (_binary_targets(y_train) and _finite_vector(f) and _finite_vector(a) and _lower_cholesky_factor(L)):
        return False
    n_samples = np.asarray(L, dtype=np.float64).shape[0]
    return bool(
        np.asarray(y_train, dtype=np.float64).shape == (n_samples,)
        and np.asarray(f, dtype=np.float64).shape == (n_samples,)
        and np.asarray(a, dtype=np.float64).shape == (n_samples,)
    )


def _finite_scalar(value: float) -> bool:
    return bool(np.isscalar(value) and np.isfinite(float(value)))


def _posterior_mean_inputs_valid(K_star: NDArray[np.float64], y_train: NDArray[np.float64], pi: NDArray[np.float64]) -> bool:
    if not (_finite_matrix(K_star) and _binary_targets(y_train) and _finite_vector(pi)):
        return False
    n_train = np.asarray(K_star, dtype=np.float64).shape[0]
    return bool(np.asarray(y_train, dtype=np.float64).shape == (n_train,) and np.asarray(pi, dtype=np.float64).shape == (n_train,))


def _same_vector_length(result: NDArray[np.float64], length: int) -> bool:
    return bool(_finite_vector(result) and np.asarray(result, dtype=np.float64).shape == (length,))


def _cross_solve_inputs_valid(L: NDArray[np.float64], W_sr: NDArray[np.float64], K_star: NDArray[np.float64]) -> bool:
    if not (_lower_cholesky_factor(L) and _finite_vector(W_sr) and _finite_matrix(K_star)):
        return False
    n_train = np.asarray(L, dtype=np.float64).shape[0]
    star = np.asarray(K_star, dtype=np.float64)
    return bool(np.asarray(W_sr, dtype=np.float64).shape == (n_train,) and star.shape[0] == n_train)


def _cross_solve_valid(result: NDArray[np.float64], L: NDArray[np.float64], K_star: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (np.asarray(L, dtype=np.float64).shape[0], np.asarray(K_star, dtype=np.float64).shape[1])
        and np.all(np.isfinite(values))
    )


def _variance_inputs_valid(kernel_diag: NDArray[np.float64], v: NDArray[np.float64]) -> bool:
    if not (_finite_vector(kernel_diag) and _finite_matrix(v)):
        return False
    return bool(np.asarray(v, dtype=np.float64).shape[1] == np.asarray(kernel_diag, dtype=np.float64).shape[0])


def _positive_vector(values: NDArray[np.float64]) -> bool:
    return bool(_finite_vector(values) and np.all(np.asarray(values, dtype=np.float64) > 0.0))


def _probability_vector(values: NDArray[np.float64]) -> bool:
    return bool(_finite_vector(values) and np.all(np.asarray(values, dtype=np.float64) >= 0.0) and np.all(np.asarray(values, dtype=np.float64) <= 1.0))


def _proba_matrix_valid(result: NDArray[np.float64], pi_star: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_test = np.asarray(pi_star, dtype=np.float64).shape[0]
    return bool(
        values.shape == (n_test, 2)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.allclose(values.sum(axis=1), 1.0)
    )


@register_atom(witness_gp_classifier_laplace_newton_step)
@icontract.require(lambda K, y_train, f: _laplace_inputs_valid(K, y_train, f), "K must be a finite symmetric positive-semidefinite matrix and y_train/f must be compatible vectors")
@icontract.ensure(lambda result, K: _laplace_step_result_valid(result, K), "Laplace Newton step must return finite compatible vectors and a lower Cholesky factor")
def gp_classifier_laplace_newton_step(
    K: NDArray[np.float64],
    y_train: NDArray[np.float64],
    f: NDArray[np.float64],
) -> LaplaceNewtonStep:
    """Compute one binary Laplace-approximation Newton step for Gaussian-process classification."""
    kernel = np.asarray(K, dtype=np.float64)
    targets = np.asarray(y_train, dtype=np.float64)
    latent = np.asarray(f, dtype=np.float64)
    pi = np.asarray(expit(latent), dtype=np.float64)
    W = pi * (1.0 - pi)
    W_sr = np.asarray(np.sqrt(W), dtype=np.float64)
    W_sr_K = W_sr[:, np.newaxis] * kernel
    B = np.eye(W.shape[0], dtype=np.float64) + W_sr_K * W_sr
    L = np.asarray(cholesky(B, lower=True, check_finite=False), dtype=np.float64)
    b = np.asarray(W * latent + (targets - pi), dtype=np.float64)
    a = np.asarray(
        b - W_sr * cho_solve((L, True), W_sr_K.dot(b), check_finite=False),
        dtype=np.float64,
    )
    f_next = np.asarray(kernel.dot(a), dtype=np.float64)
    return f_next, pi, W_sr, L, b, a


@register_atom(witness_gp_classifier_laplace_log_marginal_likelihood)
@icontract.require(lambda y_train, f, a, L: _laplace_lml_inputs_valid(y_train, f, a, L), "y_train, f, a, and L must be finite and shape-compatible")
@icontract.ensure(lambda result: _finite_scalar(result), "log-marginal likelihood must be finite")
def gp_classifier_laplace_log_marginal_likelihood(
    y_train: NDArray[np.float64],
    f: NDArray[np.float64],
    a: NDArray[np.float64],
    L: NDArray[np.float64],
) -> float:
    """Compute sklearn's binary Laplace Gaussian-process-classifier log-marginal likelihood."""
    targets = np.asarray(y_train, dtype=np.float64)
    latent = np.asarray(f, dtype=np.float64)
    alpha = np.asarray(a, dtype=np.float64)
    cholesky_factor = np.asarray(L, dtype=np.float64)
    return float(
        -0.5 * alpha.T.dot(latent)
        - np.log1p(np.exp(-(targets * 2.0 - 1.0) * latent)).sum()
        - np.log(np.diag(cholesky_factor)).sum()
    )


@register_atom(witness_gp_classifier_posterior_mean)
@icontract.require(lambda K_star, y_train, pi: _posterior_mean_inputs_valid(K_star, y_train, pi), "K_star, y_train, and pi must be finite and compatible")
@icontract.ensure(lambda result, K_star: _same_vector_length(result, np.asarray(K_star, dtype=np.float64).shape[1]), "posterior mean must match the number of test points")
def gp_classifier_posterior_mean(
    K_star: NDArray[np.float64],
    y_train: NDArray[np.float64],
    pi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute binary GP-classifier latent posterior means for test points."""
    cross_kernel = np.asarray(K_star, dtype=np.float64)
    return np.asarray(cross_kernel.T.dot(np.asarray(y_train, dtype=np.float64) - np.asarray(pi, dtype=np.float64)), dtype=np.float64)


@register_atom(witness_gp_classifier_posterior_cross_solve)
@icontract.require(lambda L, W_sr, K_star: _cross_solve_inputs_valid(L, W_sr, K_star), "L, W_sr, and K_star must be finite and shape-compatible")
@icontract.ensure(lambda result, L, K_star: _cross_solve_valid(result, L, K_star), "posterior cross solve must return a finite train-by-test matrix")
def gp_classifier_posterior_cross_solve(
    L: NDArray[np.float64],
    W_sr: NDArray[np.float64],
    K_star: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the triangular binary GP-classifier posterior cross term."""
    return np.asarray(
        solve(
            np.asarray(L, dtype=np.float64),
            np.asarray(W_sr, dtype=np.float64)[:, np.newaxis] * np.asarray(K_star, dtype=np.float64),
            check_finite=False,
        ),
        dtype=np.float64,
    )


@register_atom(witness_gp_classifier_posterior_variance)
@icontract.require(lambda kernel_diag, v: _variance_inputs_valid(kernel_diag, v), "kernel_diag and v must be finite and shape-compatible")
@icontract.ensure(lambda result, kernel_diag: _same_vector_length(result, np.asarray(kernel_diag, dtype=np.float64).shape[0]), "posterior variance must match the number of test points")
def gp_classifier_posterior_variance(
    kernel_diag: NDArray[np.float64],
    v: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute binary GP-classifier latent posterior variances for test points."""
    return np.asarray(np.asarray(kernel_diag, dtype=np.float64) - np.einsum("ij,ij->j", np.asarray(v, dtype=np.float64), np.asarray(v, dtype=np.float64)), dtype=np.float64)


@register_atom(witness_gp_classifier_predictive_probability)
@icontract.require(lambda f_star: _finite_vector(f_star), "f_star must be a finite vector")
@icontract.require(lambda var_f_star: _positive_vector(var_f_star), "var_f_star must be a strictly positive finite vector")
@icontract.require(lambda f_star, var_f_star: np.asarray(f_star, dtype=np.float64).shape == np.asarray(var_f_star, dtype=np.float64).shape, "f_star and var_f_star must have the same shape")
@icontract.ensure(lambda result, f_star: _same_vector_length(result, np.asarray(f_star, dtype=np.float64).shape[0]) and _probability_vector(result), "predictive probabilities must be finite and within [0, 1]")
def gp_classifier_predictive_probability(
    f_star: NDArray[np.float64],
    var_f_star: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Approximate binary GP-classifier predictive probabilities from latent moments."""
    latent_mean = np.asarray(f_star, dtype=np.float64)
    latent_var = np.asarray(var_f_star, dtype=np.float64)
    alpha = 1.0 / (2.0 * latent_var)
    gamma = _LAMBDAS[:, np.newaxis] * latent_mean[np.newaxis, :]
    integrals = (
        np.sqrt(np.pi / alpha)[np.newaxis, :]
        * erf(gamma * np.sqrt(alpha[np.newaxis, :] / (alpha[np.newaxis, :] + _LAMBDAS[:, np.newaxis] ** 2)))
        / (2.0 * np.sqrt(latent_var[np.newaxis, :] * 2.0 * np.pi))
    )
    return np.asarray((_COEFS[:, np.newaxis] * integrals).sum(axis=0) + 0.5 * _COEFS.sum(), dtype=np.float64)


@register_atom(witness_gp_classifier_predictive_proba)
@icontract.require(lambda pi_star: _probability_vector(pi_star), "pi_star must be a finite probability vector")
@icontract.ensure(lambda result, pi_star: _proba_matrix_valid(result, pi_star), "predictive class-probability matrix must be finite, normalized, and two-column")
def gp_classifier_predictive_proba(pi_star: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stack binary class probabilities into sklearn's two-column output format."""
    probs = np.asarray(pi_star, dtype=np.float64)
    return np.asarray(np.vstack((1.0 - probs, probs)).T, dtype=np.float64)
