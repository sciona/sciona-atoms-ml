"""Binary Gaussian-process classification post-fit attribute atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_binary_fit_L,
    witness_gpc_binary_fit_log_marginal_likelihood_value,
    witness_gpc_binary_fit_pi,
    witness_gpc_binary_fit_return_self,
    witness_gpc_binary_fit_W_sr,
)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size >= 1 and np.all(np.isfinite(array)))


def _probability_vector(values: object) -> bool:
    if not _finite_vector(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(array > 0.0) and np.all(array < 1.0))


def _nonnegative_vector(values: object) -> bool:
    return bool(_finite_vector(values) and np.all(np.asarray(values, dtype=np.float64) >= 0.0))


def _lower_cholesky_factor(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
        and np.allclose(array, np.tril(array))
        and np.all(np.diag(array) > 0.0)
    )


def _finite_float(value: object) -> bool:
    return bool(np.isscalar(value) and np.isfinite(float(value)))


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


@register_atom(witness_gpc_binary_fit_pi)
@icontract.require(lambda pi: _probability_vector(pi), "pi must be a finite probability vector with values in (0, 1)")
@icontract.ensure(
    lambda result: _probability_vector(result),
    "result must preserve a finite probability vector with values in (0, 1)",
)
def gpc_binary_fit_pi(pi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Expose _BinaryGaussianProcessClassifierLaplace.fit's fitted pi_ vector."""
    return np.asarray(pi, dtype=np.float64)


@register_atom(witness_gpc_binary_fit_W_sr)
@icontract.require(lambda W_sr: _nonnegative_vector(W_sr), "W_sr must be a finite nonnegative vector")
@icontract.ensure(
    lambda result: _nonnegative_vector(result),
    "result must preserve a finite nonnegative vector",
)
def gpc_binary_fit_W_sr(W_sr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Expose _BinaryGaussianProcessClassifierLaplace.fit's fitted W_sr_ vector."""
    return np.asarray(W_sr, dtype=np.float64)


@register_atom(witness_gpc_binary_fit_L)
@icontract.require(lambda L: _lower_cholesky_factor(L), "L must be a finite lower-triangular Cholesky factor with positive diagonal")
@icontract.ensure(
    lambda result: _lower_cholesky_factor(result),
    "result must preserve a finite lower-triangular Cholesky factor with positive diagonal",
)
def gpc_binary_fit_L(L: NDArray[np.float64]) -> NDArray[np.float64]:
    """Expose _BinaryGaussianProcessClassifierLaplace.fit's fitted L_ Cholesky factor."""
    return np.asarray(L, dtype=np.float64)


@register_atom(witness_gpc_binary_fit_log_marginal_likelihood_value)
@icontract.require(
    lambda log_marginal_likelihood_value: _finite_float(log_marginal_likelihood_value),
    "log_marginal_likelihood_value must be finite",
)
@icontract.ensure(
    lambda result: _finite_float(result),
    "result must be a finite scalar",
)
def gpc_binary_fit_log_marginal_likelihood_value(
    log_marginal_likelihood_value: float,
) -> float:
    """Expose _BinaryGaussianProcessClassifierLaplace.fit's fitted log_marginal_likelihood_value_."""
    return float(log_marginal_likelihood_value)


@register_atom(witness_gpc_binary_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(lambda result, estimator_token: result == estimator_token, "fit return value must preserve the estimator token")
def gpc_binary_fit_return_self(estimator_token: str) -> str:
    """Model _BinaryGaussianProcessClassifierLaplace.fit returning self after fitted-state updates."""
    return estimator_token
