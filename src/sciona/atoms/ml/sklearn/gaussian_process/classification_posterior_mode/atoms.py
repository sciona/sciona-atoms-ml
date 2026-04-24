"""Gaussian-process classification posterior-mode atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from ..classification.atoms import (
    gp_classifier_laplace_log_marginal_likelihood,
    gp_classifier_laplace_newton_step,
)
from .witnesses import (
    witness_gp_classifier_posterior_mode,
    witness_gp_classifier_posterior_mode_converged,
    witness_gp_classifier_posterior_mode_initial_latent,
)

PosteriorModeState = tuple[
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _binary_targets(values: object) -> bool:
    if not _finite_vector(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isin(array, np.array([0.0, 1.0], dtype=np.float64))))


def _finite_square_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[0] == array.shape[1] and np.all(np.isfinite(array)))


def _symmetric_positive_semidefinite(values: object) -> bool:
    if not _finite_square_matrix(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    if not np.allclose(array, array.T):
        return False
    return bool(np.all(np.linalg.eigvalsh(array) >= -1e-10))


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _finite_or_negative_infinity(value: float) -> bool:
    if isinstance(value, bool) or not np.isscalar(value):
        return False
    scalar = float(value)
    return bool(np.isfinite(scalar) or scalar == float("-inf"))


def _positive_finite(value: float) -> bool:
    return bool(not isinstance(value, bool) and np.isscalar(value) and np.isfinite(float(value)) and float(value) > 0.0)


def _cached_latent_valid(y_train: NDArray[np.float64], cached_f: NDArray[np.float64] | None) -> bool:
    if cached_f is None:
        return True
    return bool(_finite_vector(cached_f))


def _same_length_float64(result: NDArray[np.float64], length: int) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (length,) and values.dtype == np.float64 and np.all(np.isfinite(values)))


def _posterior_mode_inputs_valid(
    K: NDArray[np.float64],
    y_train: NDArray[np.float64],
    max_iter_predict: int,
    cached_f: NDArray[np.float64] | None,
) -> bool:
    if not (_symmetric_positive_semidefinite(K) and _binary_targets(y_train) and _positive_int(max_iter_predict)):
        return False
    kernel = np.asarray(K, dtype=np.float64)
    targets = np.asarray(y_train, dtype=np.float64)
    return bool(targets.shape == (kernel.shape[0],) and _cached_latent_valid(targets, cached_f))


def _lower_cholesky_factor(values: NDArray[np.float64]) -> bool:
    if not _finite_square_matrix(values):
        return False
    factor = np.asarray(values, dtype=np.float64)
    return bool(np.allclose(factor, np.tril(factor)) and np.all(np.diag(factor) > 0.0))


def _posterior_mode_result_valid(result: PosteriorModeState, K: NDArray[np.float64]) -> bool:
    if not isinstance(result, tuple) or len(result) != 7:
        return False
    lml, f_cached, pi, w_sr, L, b, a = result
    n_samples = np.asarray(K, dtype=np.float64).shape[0]
    return bool(
        _finite_or_negative_infinity(lml)
        and _same_length_float64(f_cached, n_samples)
        and _same_length_float64(pi, n_samples)
        and _same_length_float64(w_sr, n_samples)
        and _same_length_float64(b, n_samples)
        and _same_length_float64(a, n_samples)
        and _lower_cholesky_factor(L)
        and np.asarray(L, dtype=np.float64).shape == (n_samples, n_samples)
    )


@register_atom(witness_gp_classifier_posterior_mode_initial_latent)
@icontract.require(lambda y_train: _binary_targets(y_train), "y_train must be a finite binary vector")
@icontract.require(lambda y_train, cached_f: _cached_latent_valid(y_train, cached_f), "cached_f must be None or a finite vector matching y_train")
@icontract.ensure(lambda result, y_train: _same_length_float64(result, np.asarray(y_train, dtype=np.float64).shape[0]), "initial latent vector must be finite float64 with the same length as y_train")
def gp_classifier_posterior_mode_initial_latent(
    y_train: NDArray[np.float64],
    *,
    warm_start: bool = False,
    cached_f: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Choose the starting latent vector for sklearn's binary Laplace posterior-mode loop."""
    targets = np.asarray(y_train, dtype=np.float64)
    if warm_start and cached_f is not None and np.asarray(cached_f, dtype=np.float64).shape == targets.shape:
        return np.asarray(cached_f, dtype=np.float64).copy()
    return np.zeros_like(targets, dtype=np.float64)


@register_atom(witness_gp_classifier_posterior_mode_converged)
@icontract.require(lambda previous_log_marginal_likelihood: _finite_or_negative_infinity(previous_log_marginal_likelihood), "previous_log_marginal_likelihood must be finite or negative infinity")
@icontract.require(lambda current_log_marginal_likelihood: np.isfinite(float(current_log_marginal_likelihood)), "current_log_marginal_likelihood must be finite")
@icontract.require(lambda tolerance: _positive_finite(tolerance), "tolerance must be positive and finite")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def gp_classifier_posterior_mode_converged(
    previous_log_marginal_likelihood: float,
    current_log_marginal_likelihood: float,
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Check sklearn's binary Laplace posterior-mode improvement criterion."""
    previous = float(previous_log_marginal_likelihood)
    current = float(current_log_marginal_likelihood)
    return bool(current - previous < float(tolerance))


@register_atom(witness_gp_classifier_posterior_mode)
@icontract.require(
    lambda K, y_train, max_iter_predict, cached_f: _posterior_mode_inputs_valid(K, y_train, max_iter_predict, cached_f),
    "K, y_train, max_iter_predict, and cached_f must be finite and shape-compatible",
)
@icontract.ensure(lambda result, K: _posterior_mode_result_valid(result, K), "posterior-mode solve must return finite latent state, Newton temporaries, and log-marginal likelihood")
def gp_classifier_posterior_mode(
    K: NDArray[np.float64],
    y_train: NDArray[np.float64],
    *,
    max_iter_predict: int = 100,
    warm_start: bool = False,
    cached_f: NDArray[np.float64] | None = None,
) -> PosteriorModeState:
    """Run sklearn's fixed-kernel binary Laplace posterior-mode Newton loop."""
    kernel = np.asarray(K, dtype=np.float64)
    targets = np.asarray(y_train, dtype=np.float64)
    f = gp_classifier_posterior_mode_initial_latent(
        targets,
        warm_start=warm_start,
        cached_f=cached_f,
    )

    log_marginal_likelihood = float("-inf")
    pi = np.zeros_like(f, dtype=np.float64)
    w_sr = np.zeros_like(f, dtype=np.float64)
    L = np.eye(f.shape[0], dtype=np.float64)
    b = np.zeros_like(f, dtype=np.float64)
    a = np.zeros_like(f, dtype=np.float64)

    for _ in range(int(max_iter_predict)):
        f, pi, w_sr, L, b, a = gp_classifier_laplace_newton_step(kernel, targets, f)
        lml = gp_classifier_laplace_log_marginal_likelihood(targets, f, a, L)
        if gp_classifier_posterior_mode_converged(log_marginal_likelihood, lml):
            log_marginal_likelihood = float(lml)
            break
        log_marginal_likelihood = float(lml)

    return (
        float(log_marginal_likelihood),
        np.asarray(f, dtype=np.float64),
        np.asarray(pi, dtype=np.float64),
        np.asarray(w_sr, dtype=np.float64),
        np.asarray(L, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.asarray(a, dtype=np.float64),
    )
