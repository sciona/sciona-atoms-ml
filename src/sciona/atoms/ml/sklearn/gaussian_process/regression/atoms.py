"""Gaussian-process regression linear-algebra atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_solve, cholesky, solve_triangular

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_dual_coefficients,
    witness_gp_log_marginal_likelihood,
    witness_gp_posterior_cross_solve,
    witness_gp_posterior_predictive_covariance,
    witness_gp_posterior_predictive_mean,
    witness_gp_posterior_predictive_std,
    witness_gp_regularized_train_kernel,
    witness_gp_train_cholesky,
)

AlphaLike = float | NDArray[np.float64]
TargetStats = float | NDArray[np.float64]


def _finite_square_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] == array.shape[1] and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim in {1, 2} and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _alpha_valid(alpha: AlphaLike, n_samples: int) -> bool:
    if isinstance(alpha, bool):
        return False
    if np.isscalar(alpha):
        return bool(np.isfinite(float(alpha)) and float(alpha) >= 0.0)
    try:
        values = np.asarray(alpha, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape == (n_samples,) and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _regularized_kernel_valid(result: NDArray[np.float64], K: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(K, dtype=np.float64)
    return bool(values.shape == source.shape and np.all(np.isfinite(values)) and np.allclose(values, values.T))


def _symmetric_positive_definite(values: NDArray[np.float64]) -> bool:
    if not _finite_square_matrix(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    if not np.allclose(array, array.T):
        return False
    try:
        np.linalg.cholesky(array)
    except np.linalg.LinAlgError:
        return False
    return True


def _cholesky_valid(result: NDArray[np.float64], K: NDArray[np.float64]) -> bool:
    factor = np.asarray(result, dtype=np.float64)
    source = np.asarray(K, dtype=np.float64)
    return bool(
        factor.shape == source.shape
        and np.all(np.isfinite(factor))
        and np.allclose(factor, np.tril(factor))
        and np.all(np.diag(factor) > 0.0)
        and np.allclose(factor @ factor.T, source)
    )


def _lower_cholesky_factor(values: NDArray[np.float64]) -> bool:
    try:
        factor = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        factor.ndim == 2
        and factor.shape[0] == factor.shape[1]
        and factor.shape[0] >= 1
        and np.all(np.isfinite(factor))
        and np.allclose(factor, np.tril(factor))
        and np.all(np.diag(factor) > 0.0)
    )


def _targets_compatible(L: NDArray[np.float64], y_train: NDArray[np.float64]) -> bool:
    if not (_lower_cholesky_factor(L) and _finite_matrix(y_train)):
        return False
    y_values = np.asarray(y_train, dtype=np.float64)
    return bool(y_values.shape[0] == np.asarray(L).shape[0])


def _same_shape(result: NDArray[np.float64], source: NDArray[np.float64]) -> bool:
    return bool(np.asarray(result).shape == np.asarray(source).shape and np.all(np.isfinite(result)))


def _lml_inputs_valid(y_train: NDArray[np.float64], dual_coefficients: NDArray[np.float64], L: NDArray[np.float64]) -> bool:
    return bool(_targets_compatible(L, y_train) and _finite_matrix(dual_coefficients) and np.asarray(dual_coefficients).shape == np.asarray(y_train).shape)


def _finite_scalar(value: float) -> bool:
    return bool(np.isscalar(value) and np.isfinite(float(value)))


def _k_trans_valid(K_trans: NDArray[np.float64], dual_coefficients: NDArray[np.float64]) -> bool:
    if not (_finite_matrix(K_trans) and _finite_matrix(dual_coefficients)):
        return False
    k_values = np.asarray(K_trans, dtype=np.float64)
    alpha_values = np.asarray(dual_coefficients, dtype=np.float64)
    return bool(k_values.ndim == 2 and alpha_values.ndim in {1, 2} and k_values.shape[1] == alpha_values.shape[0])


def _target_stats_valid(value: TargetStats, n_targets: int) -> bool:
    if isinstance(value, bool):
        return False
    if np.isscalar(value):
        return bool(np.isfinite(float(value)))
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape == (n_targets,) and np.all(np.isfinite(values)))


def _target_scale_valid(value: TargetStats, n_targets: int) -> bool:
    if not _target_stats_valid(value, n_targets):
        return False
    return bool(np.all(np.asarray(value, dtype=np.float64) > 0.0))


def _posterior_mean_inputs_valid(
    K_trans: NDArray[np.float64],
    dual_coefficients: NDArray[np.float64],
    y_train_mean: TargetStats,
    y_train_std: TargetStats,
) -> bool:
    if not _k_trans_valid(K_trans, dual_coefficients):
        return False
    alpha_values = np.asarray(dual_coefficients, dtype=np.float64)
    n_targets = 1 if alpha_values.ndim == 1 else alpha_values.shape[1]
    return bool(_target_stats_valid(y_train_mean, n_targets) and _target_scale_valid(y_train_std, n_targets))


def _posterior_mean_valid(result: NDArray[np.float64], K_trans: NDArray[np.float64], dual_coefficients: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    alpha_values = np.asarray(dual_coefficients, dtype=np.float64)
    n_test = np.asarray(K_trans).shape[0]
    if alpha_values.ndim == 1 or alpha_values.shape[1] == 1:
        expected_shape = (n_test,)
    else:
        expected_shape = (n_test, alpha_values.shape[1])
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _cross_solve_inputs_valid(L: NDArray[np.float64], K_trans: NDArray[np.float64]) -> bool:
    if not (_lower_cholesky_factor(L) and _finite_matrix(K_trans)):
        return False
    k_values = np.asarray(K_trans, dtype=np.float64)
    return bool(k_values.ndim == 2 and k_values.shape[1] == np.asarray(L).shape[0])


def _cross_solve_valid(result: NDArray[np.float64], L: NDArray[np.float64], K_trans: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (np.asarray(L).shape[0], np.asarray(K_trans).shape[0]) and np.all(np.isfinite(values)))


def _covariance_inputs_valid(K_test: NDArray[np.float64], V: NDArray[np.float64], y_train_std: TargetStats) -> bool:
    if not (_finite_square_matrix(K_test) and _finite_matrix(V)):
        return False
    v_values = np.asarray(V, dtype=np.float64)
    n_test = np.asarray(K_test).shape[0]
    return bool(v_values.ndim == 2 and v_values.shape[1] == n_test and _target_scale_valid(y_train_std, 1 if np.isscalar(y_train_std) else np.asarray(y_train_std).shape[0]))


def _covariance_valid(result: NDArray[np.float64], K_test: NDArray[np.float64], y_train_std: TargetStats) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_test = np.asarray(K_test).shape[0]
    n_targets = 1 if np.isscalar(y_train_std) else np.asarray(y_train_std).shape[0]
    expected_shape = (n_test, n_test) if n_targets == 1 else (n_test, n_test, n_targets)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _std_inputs_valid(kernel_diag: NDArray[np.float64], V: NDArray[np.float64], y_train_std: TargetStats) -> bool:
    if not (_finite_matrix(kernel_diag) and _finite_matrix(V)):
        return False
    diag_values = np.asarray(kernel_diag, dtype=np.float64)
    v_values = np.asarray(V, dtype=np.float64)
    return bool(diag_values.ndim == 1 and v_values.ndim == 2 and v_values.shape[1] == diag_values.shape[0] and _target_scale_valid(y_train_std, 1 if np.isscalar(y_train_std) else np.asarray(y_train_std).shape[0]))


def _std_valid(result: NDArray[np.float64], kernel_diag: NDArray[np.float64], y_train_std: TargetStats) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_test = np.asarray(kernel_diag).shape[0]
    n_targets = 1 if np.isscalar(y_train_std) else np.asarray(y_train_std).shape[0]
    expected_shape = (n_test,) if n_targets == 1 else (n_test, n_targets)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)) and np.all(values >= 0.0))


@register_atom(witness_gp_regularized_train_kernel)
@icontract.require(lambda K, alpha: _finite_square_matrix(K) and _alpha_valid(alpha, np.asarray(K).shape[0]), "K must be finite square and alpha must be scalar or sample-length nonnegative vector")
@icontract.ensure(lambda result, K: _regularized_kernel_valid(result, K), "regularized kernel must remain finite, square, and symmetric")
def gp_regularized_train_kernel(K: NDArray[np.float64], alpha: AlphaLike) -> NDArray[np.float64]:
    """Add sklearn GaussianProcessRegressor alpha noise to the kernel diagonal."""
    values = np.asarray(K, dtype=np.float64).copy()
    values[np.diag_indices_from(values)] += alpha
    return values


@register_atom(witness_gp_train_cholesky)
@icontract.require(lambda K_regularized: _symmetric_positive_definite(K_regularized), "regularized kernel must be symmetric positive definite")
@icontract.ensure(lambda result, K_regularized: _cholesky_valid(result, K_regularized), "Cholesky factor must reconstruct the regularized kernel")
def gp_train_cholesky(K_regularized: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the lower Cholesky factor used by Gaussian-process regression."""
    return np.asarray(cholesky(np.asarray(K_regularized, dtype=np.float64), lower=True, check_finite=False), dtype=np.float64)


@register_atom(witness_gp_dual_coefficients)
@icontract.require(lambda L, y_train: _targets_compatible(L, y_train), "training targets must match the Cholesky sample count")
@icontract.ensure(lambda result, y_train: _same_shape(result, y_train), "dual coefficients must match the target shape")
def gp_dual_coefficients(L: NDArray[np.float64], y_train: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve for the dual coefficients used by Gaussian-process predictions."""
    return np.asarray(cho_solve((np.asarray(L, dtype=np.float64), True), np.asarray(y_train, dtype=np.float64), check_finite=False), dtype=np.float64)


@register_atom(witness_gp_log_marginal_likelihood)
@icontract.require(lambda y_train, dual_coefficients, L: _lml_inputs_valid(y_train, dual_coefficients, L), "targets, dual coefficients, and Cholesky factor must be compatible")
@icontract.ensure(lambda result: _finite_scalar(result), "log-marginal likelihood must be finite")
def gp_log_marginal_likelihood(
    y_train: NDArray[np.float64],
    dual_coefficients: NDArray[np.float64],
    L: NDArray[np.float64],
) -> float:
    """Compute sklearn's GaussianProcessRegressor log-marginal likelihood."""
    y_values = np.asarray(y_train, dtype=np.float64)
    alpha_values = np.asarray(dual_coefficients, dtype=np.float64)
    if y_values.ndim == 1:
        y_values = y_values[:, np.newaxis]
        alpha_values = alpha_values[:, np.newaxis]
    dims = -0.5 * np.einsum("ik,ik->k", y_values, alpha_values)
    dims -= np.log(np.diag(np.asarray(L, dtype=np.float64))).sum()
    dims -= np.asarray(L).shape[0] / 2.0 * np.log(2.0 * np.pi)
    return float(dims.sum(axis=-1))


@register_atom(witness_gp_posterior_predictive_mean)
@icontract.require(lambda K_trans, dual_coefficients, y_train_mean, y_train_std: _posterior_mean_inputs_valid(K_trans, dual_coefficients, y_train_mean, y_train_std), "cross-kernel, dual coefficients, and target normalization stats must be compatible")
@icontract.ensure(lambda result, K_trans, dual_coefficients: _posterior_mean_valid(result, K_trans, dual_coefficients), "posterior mean must match test and target dimensions")
def gp_posterior_predictive_mean(
    K_trans: NDArray[np.float64],
    dual_coefficients: NDArray[np.float64],
    y_train_mean: TargetStats = 0.0,
    y_train_std: TargetStats = 1.0,
) -> NDArray[np.float64]:
    """Compute the Gaussian-process posterior predictive mean."""
    mean = np.asarray(K_trans, dtype=np.float64) @ np.asarray(dual_coefficients, dtype=np.float64)
    mean = np.asarray(y_train_std, dtype=np.float64) * mean + np.asarray(y_train_mean, dtype=np.float64)
    if mean.ndim > 1 and mean.shape[1] == 1:
        mean = np.squeeze(mean, axis=1)
    return np.asarray(mean, dtype=np.float64)


@register_atom(witness_gp_posterior_cross_solve)
@icontract.require(lambda L, K_trans: _cross_solve_inputs_valid(L, K_trans), "cross-kernel columns must match the Cholesky sample count")
@icontract.ensure(lambda result, L, K_trans: _cross_solve_valid(result, L, K_trans), "posterior cross solve must have train-by-test shape")
def gp_posterior_cross_solve(L: NDArray[np.float64], K_trans: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve the triangular posterior cross term V = L \\ K_trans.T."""
    return np.asarray(solve_triangular(np.asarray(L, dtype=np.float64), np.asarray(K_trans, dtype=np.float64).T, lower=True, check_finite=False), dtype=np.float64)


@register_atom(witness_gp_posterior_predictive_covariance)
@icontract.require(lambda K_test, V, y_train_std: _covariance_inputs_valid(K_test, V, y_train_std), "test kernel, cross solve, and target scale must be compatible")
@icontract.ensure(lambda result, K_test, y_train_std: _covariance_valid(result, K_test, y_train_std), "posterior covariance must match test and target dimensions")
def gp_posterior_predictive_covariance(
    K_test: NDArray[np.float64],
    V: NDArray[np.float64],
    y_train_std: TargetStats = 1.0,
) -> NDArray[np.float64]:
    """Compute the Gaussian-process posterior predictive covariance."""
    covariance = np.asarray(K_test, dtype=np.float64) - np.asarray(V, dtype=np.float64).T @ np.asarray(V, dtype=np.float64)
    covariance = np.outer(covariance, np.asarray(y_train_std, dtype=np.float64) ** 2).reshape(*covariance.shape, -1)
    if covariance.shape[2] == 1:
        covariance = np.squeeze(covariance, axis=2)
    return np.asarray(covariance, dtype=np.float64)


@register_atom(witness_gp_posterior_predictive_std)
@icontract.require(lambda kernel_diag, V, y_train_std: _std_inputs_valid(kernel_diag, V, y_train_std), "kernel diagonal, cross solve, and target scale must be compatible")
@icontract.ensure(lambda result, kernel_diag, y_train_std: _std_valid(result, kernel_diag, y_train_std), "posterior standard deviation must be finite and nonnegative")
def gp_posterior_predictive_std(
    kernel_diag: NDArray[np.float64],
    V: NDArray[np.float64],
    y_train_std: TargetStats = 1.0,
) -> NDArray[np.float64]:
    """Compute the Gaussian-process posterior predictive standard deviation."""
    variance = np.asarray(kernel_diag, dtype=np.float64).copy()
    variance -= np.einsum("ij,ji->i", np.asarray(V, dtype=np.float64).T, np.asarray(V, dtype=np.float64))
    variance[variance < 0.0] = 0.0
    variance = np.outer(variance, np.asarray(y_train_std, dtype=np.float64) ** 2).reshape(*variance.shape, -1)
    if variance.shape[1] == 1:
        variance = np.squeeze(variance, axis=1)
    return np.asarray(np.sqrt(variance), dtype=np.float64)
