"""Ghost witnesses for binary Gaussian-process classification post-fit attribute atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def witness_gpc_binary_fit_pi(pi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Describe the fitted binary GPC positive-class training probabilities."""
    return np.asarray(pi, dtype=np.float64)


def witness_gpc_binary_fit_W_sr(W_sr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Describe the fitted binary GPC square-root Hessian diagonal."""
    return np.asarray(W_sr, dtype=np.float64)


def witness_gpc_binary_fit_L(L: NDArray[np.float64]) -> NDArray[np.float64]:
    """Describe the fitted binary GPC lower Cholesky factor."""
    return np.asarray(L, dtype=np.float64)


def witness_gpc_binary_fit_log_marginal_likelihood_value(
    log_marginal_likelihood_value: float,
) -> float:
    """Describe the fitted binary GPC log-marginal-likelihood scalar."""
    return float(log_marginal_likelihood_value)


def witness_gpc_binary_fit_return_self(estimator_token: str) -> str:
    """Describe the fit self-return passthrough."""
    return estimator_token
