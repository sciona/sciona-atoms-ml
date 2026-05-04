"""Ghost witnesses for Gaussian-process classification fit-state shell atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def witness_gpc_fit_binary_base_estimator(
    kernel: object,
    optimizer: object,
    n_restarts_optimizer: int,
    max_iter_predict: int,
    warm_start: bool,
    copy_X_train: bool,
    random_state: object,
) -> str:
    """Describe binary base-estimator construction during GaussianProcessClassifier.fit."""
    del kernel, optimizer, n_restarts_optimizer, max_iter_predict, warm_start, copy_X_train, random_state
    return "_BinaryGaussianProcessClassifierLaplace"


def witness_gpc_fit_one_vs_rest_estimator(
    base_estimator_token: str,
    n_jobs: int | None,
) -> str:
    """Describe the one-vs-rest wrapper branch during GaussianProcessClassifier.fit."""
    del base_estimator_token, n_jobs
    return "OneVsRestClassifier"


def witness_gpc_fit_one_vs_one_estimator(
    base_estimator_token: str,
    n_jobs: int | None,
) -> str:
    """Describe the one-vs-one wrapper branch during GaussianProcessClassifier.fit."""
    del base_estimator_token, n_jobs
    return "OneVsOneClassifier"


def witness_gpc_fit_binary_log_marginal_likelihood_value(
    base_estimator_log_marginal_likelihood_value: float,
) -> float:
    """Describe the binary fitted log-marginal-likelihood attribute copy."""
    return float(base_estimator_log_marginal_likelihood_value)


def witness_gpc_fit_multiclass_log_marginal_likelihood_value(
    estimator_log_marginal_likelihood_values: NDArray[np.float64],
) -> float:
    """Describe the multiclass fitted log-marginal-likelihood mean aggregation."""
    return float(np.mean(np.asarray(estimator_log_marginal_likelihood_values, dtype=np.float64)))


def witness_gpc_fit_return_self(estimator_token: str) -> str:
    """Describe the fit self-return passthrough."""
    return estimator_token
