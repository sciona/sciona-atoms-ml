"""Ghost witnesses for binary Gaussian-process classification predict_proba output atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf


def witness_gpc_binary_predict_proba_alpha(
    var_f_star: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the positive alpha scaling vector derived from predictive variances."""
    return 1.0 / (2.0 * np.asarray(var_f_star, dtype=np.float64))


def witness_gpc_binary_predict_proba_gamma(
    f_star: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the five-row gamma matrix derived from latent predictive means."""
    lambdas = np.asarray([[0.41], [0.4], [0.37], [0.44], [0.39]], dtype=np.float64)
    return lambdas * np.asarray(f_star, dtype=np.float64)


def witness_gpc_binary_predict_proba_integrals(
    alpha: NDArray[np.float64],
    gamma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the five-row error-function integral approximation matrix."""
    alpha_vector = np.asarray(alpha, dtype=np.float64)
    gamma_matrix = np.asarray(gamma, dtype=np.float64)
    lambdas = np.asarray([[0.41], [0.4], [0.37], [0.44], [0.39]], dtype=np.float64)
    var_f_star = 1.0 / (2.0 * alpha_vector)
    return (
        np.sqrt(np.pi / alpha_vector)
        * erf(gamma_matrix * np.sqrt(alpha_vector / (alpha_vector + lambdas**2)))
        / (2.0 * np.sqrt(var_f_star * 2.0 * np.pi))
    )


def witness_gpc_binary_predict_proba_positive_class_probabilities(
    integrals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the binary positive-class probability vector assembled from integral rows."""
    coefs = np.asarray(
        [[-1854.8214151], [3516.89893646], [221.29346712], [128.12323805], [-2010.49422654]],
        dtype=np.float64,
    )
    integral_matrix = np.asarray(integrals, dtype=np.float64)
    return (coefs * integral_matrix).sum(axis=0) + 0.5 * coefs.sum()


def witness_gpc_binary_predict_proba_matrix(
    pi_star: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the final two-column binary probability matrix."""
    probability_vector = np.asarray(pi_star, dtype=np.float64)
    return np.vstack((1.0 - probability_vector, probability_vector)).T
