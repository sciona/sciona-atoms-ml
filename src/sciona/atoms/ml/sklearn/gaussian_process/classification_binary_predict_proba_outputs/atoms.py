"""Binary Gaussian-process classification predict_proba output atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_binary_predict_proba_alpha,
    witness_gpc_binary_predict_proba_gamma,
    witness_gpc_binary_predict_proba_integrals,
    witness_gpc_binary_predict_proba_matrix,
    witness_gpc_binary_predict_proba_positive_class_probabilities,
)

LAMBDAS = np.asarray([[0.41], [0.4], [0.37], [0.44], [0.39]], dtype=np.float64)
COEFS = np.asarray(
    [[-1854.8214151], [3516.89893646], [221.29346712], [128.12323805], [-2010.49422654]],
    dtype=np.float64,
)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size >= 1 and np.all(np.isfinite(array)))


def _positive_vector(values: object) -> bool:
    return bool(_finite_vector(values) and np.all(np.asarray(values, dtype=np.float64) > 0.0))


def _probability_vector(values: object) -> bool:
    return bool(_finite_vector(values) and np.all(np.asarray(values, dtype=np.float64) >= 0.0) and np.all(np.asarray(values, dtype=np.float64) <= 1.0))


def _five_row_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] == 5 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _aligned_gamma(alpha: object, gamma: object) -> bool:
    return bool(_positive_vector(alpha) and _five_row_matrix(gamma) and np.asarray(gamma, dtype=np.float64).shape[1] == np.asarray(alpha, dtype=np.float64).shape[0])


def _probability_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[1] == 2
        and array.shape[0] >= 1
        and np.all(np.isfinite(array))
        and np.all(array >= 0.0)
        and np.all(array <= 1.0)
        and np.allclose(array.sum(axis=1), 1.0)
    )


@register_atom(witness_gpc_binary_predict_proba_alpha)
@icontract.require(lambda var_f_star: _positive_vector(var_f_star), "var_f_star must be a finite positive one-dimensional variance vector")
@icontract.ensure(
    lambda result, var_f_star: _positive_vector(result) and np.asarray(result).shape == np.asarray(var_f_star).shape,
    "alpha must be a finite positive vector aligned with var_f_star",
)
def gpc_binary_predict_proba_alpha(
    var_f_star: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute sklearn's binary Gaussian-process predict_proba alpha vector from predictive variances."""
    return 1.0 / (2.0 * np.asarray(var_f_star, dtype=np.float64))


@register_atom(witness_gpc_binary_predict_proba_gamma)
@icontract.require(lambda f_star: _finite_vector(f_star), "f_star must be a finite one-dimensional latent-mean vector")
@icontract.ensure(
    lambda result, f_star: _five_row_matrix(result) and np.asarray(result).shape[1] == np.asarray(f_star).shape[0],
    "gamma must be a five-row finite matrix aligned with f_star",
)
def gpc_binary_predict_proba_gamma(
    f_star: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute sklearn's binary Gaussian-process predict_proba gamma matrix from latent means."""
    return LAMBDAS * np.asarray(f_star, dtype=np.float64)


@register_atom(witness_gpc_binary_predict_proba_integrals)
@icontract.require(lambda alpha: _positive_vector(alpha), "alpha must be a finite positive one-dimensional vector")
@icontract.require(lambda alpha, gamma: _aligned_gamma(alpha, gamma), "gamma must be a finite five-row matrix aligned with alpha")
@icontract.ensure(
    lambda result, alpha: _five_row_matrix(result) and np.asarray(result).shape[1] == np.asarray(alpha).shape[0],
    "integrals must be a finite five-row matrix aligned with alpha",
)
def gpc_binary_predict_proba_integrals(
    alpha: NDArray[np.float64],
    gamma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute sklearn's five-row Gaussian integral approximation matrix for binary predict_proba."""
    alpha_vector = np.asarray(alpha, dtype=np.float64)
    gamma_matrix = np.asarray(gamma, dtype=np.float64)
    var_f_star = 1.0 / (2.0 * alpha_vector)
    return (
        np.sqrt(np.pi / alpha_vector)
        * erf(gamma_matrix * np.sqrt(alpha_vector / (alpha_vector + LAMBDAS**2)))
        / (2.0 * np.sqrt(var_f_star * 2.0 * np.pi))
    )


@register_atom(witness_gpc_binary_predict_proba_positive_class_probabilities)
@icontract.require(lambda integrals: _five_row_matrix(integrals), "integrals must be a finite five-row matrix")
@icontract.ensure(
    lambda result, integrals: _probability_vector(result) and np.asarray(result).shape == (np.asarray(integrals).shape[1],),
    "positive-class probabilities must be a finite probability vector aligned with the integral columns",
)
def gpc_binary_predict_proba_positive_class_probabilities(
    integrals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble sklearn's binary positive-class probability vector from the integral approximation rows."""
    integral_matrix = np.asarray(integrals, dtype=np.float64)
    return (COEFS * integral_matrix).sum(axis=0) + 0.5 * COEFS.sum()


@register_atom(witness_gpc_binary_predict_proba_matrix)
@icontract.require(lambda pi_star: _probability_vector(pi_star), "pi_star must be a finite probability vector")
@icontract.ensure(
    lambda result, pi_star: _probability_matrix(result) and np.asarray(result).shape[0] == np.asarray(pi_star).shape[0],
    "probability matrix must be a finite two-column matrix aligned with pi_star",
)
def gpc_binary_predict_proba_matrix(
    pi_star: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build sklearn's final two-column binary probability matrix from positive-class probabilities."""
    probability_vector = np.asarray(pi_star, dtype=np.float64)
    return np.vstack((1.0 - probability_vector, probability_vector)).T
