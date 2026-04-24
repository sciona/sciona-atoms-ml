"""Gaussian-process regression prior-prediction atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_regression_prior_covariance,
    witness_gp_regression_prior_mean,
    witness_gp_regression_prior_std,
    witness_gp_regression_prior_target_count,
    witness_gp_regression_prior_variance,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _optional_positive_int(value: int | None) -> bool:
    return bool(value is None or _positive_int(value))


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_square_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
    )


def _nonnegative_vector_or_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim in {1, 2}
        and array.shape[0] >= 1
        and (array.ndim == 1 or array.shape[1] >= 1)
        and np.all(np.isfinite(array))
        and np.all(array >= 0.0)
    )


def _prior_mean_valid(result: NDArray[np.float64], n_samples: int, n_targets: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    if int(n_targets) == 1:
        return bool(values.shape == (int(n_samples),) and np.all(values == 0.0))
    return bool(values.shape == (int(n_samples), int(n_targets)) and np.all(values == 0.0))


def _prior_covariance_valid(result: NDArray[np.float64], kernel_covariance: NDArray[np.float64], n_targets: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    kernel = np.asarray(kernel_covariance, dtype=np.float64)
    if int(n_targets) == 1:
        return bool(values.shape == kernel.shape and np.array_equal(values, kernel))
    expected = np.repeat(np.expand_dims(kernel, -1), repeats=int(n_targets), axis=-1)
    return bool(values.shape == expected.shape and np.array_equal(values, expected))


def _prior_variance_valid(result: NDArray[np.float64], kernel_variance: NDArray[np.float64], n_targets: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    kernel = np.asarray(kernel_variance, dtype=np.float64)
    if int(n_targets) == 1:
        return bool(values.shape == kernel.shape and np.array_equal(values, kernel))
    expected = np.repeat(np.expand_dims(kernel, -1), repeats=int(n_targets), axis=-1)
    return bool(values.shape == expected.shape and np.array_equal(values, expected))


def _prior_std_valid(result: NDArray[np.float64], prior_variance: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    variance = np.asarray(prior_variance, dtype=np.float64)
    return bool(values.shape == variance.shape and np.allclose(values, np.sqrt(variance)))


@register_atom(witness_gp_regression_prior_target_count)
@icontract.require(lambda n_targets=None: _optional_positive_int(n_targets), "n_targets must be None or a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "resolved prior target count must be positive")
def gp_regression_prior_target_count(
    *,
    n_targets: int | None = None,
) -> int:
    """Resolve the prior-prediction target count used by unfitted GaussianProcessRegressor.predict."""
    return int(n_targets) if n_targets is not None else 1


@register_atom(witness_gp_regression_prior_mean)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda n_targets=1: _positive_int(n_targets), "n_targets must be a positive integer")
@icontract.ensure(lambda result, n_samples, n_targets=1: _prior_mean_valid(result, n_samples, n_targets), "prior mean must be a zero array with sklearn's squeezed output shape")
def gp_regression_prior_mean(
    n_samples: int,
    *,
    n_targets: int = 1,
) -> NDArray[np.float64]:
    """Construct the unfitted Gaussian-process prior mean with sklearn's target-axis squeezing."""
    return np.zeros(shape=(int(n_samples), int(n_targets)), dtype=np.float64).squeeze()


@register_atom(witness_gp_regression_prior_covariance)
@icontract.require(lambda kernel_covariance: _finite_square_matrix(kernel_covariance), "kernel_covariance must be a finite nonempty square matrix")
@icontract.require(lambda n_targets=1: _positive_int(n_targets), "n_targets must be a positive integer")
@icontract.ensure(lambda result, kernel_covariance, n_targets=1: _prior_covariance_valid(result, kernel_covariance, n_targets), "prior covariance must preserve the kernel covariance or repeat it across targets")
def gp_regression_prior_covariance(
    kernel_covariance: NDArray[np.float64],
    *,
    n_targets: int = 1,
) -> NDArray[np.float64]:
    """Format unfitted Gaussian-process prior covariance outputs from a supplied kernel covariance matrix."""
    covariance = np.asarray(kernel_covariance, dtype=np.float64)
    if int(n_targets) == 1:
        return covariance
    return np.asarray(
        np.repeat(np.expand_dims(covariance, -1), repeats=int(n_targets), axis=-1),
        dtype=np.float64,
    )


@register_atom(witness_gp_regression_prior_variance)
@icontract.require(lambda kernel_variance: _finite_vector(kernel_variance), "kernel_variance must be a finite nonempty vector")
@icontract.require(lambda n_targets=1: _positive_int(n_targets), "n_targets must be a positive integer")
@icontract.ensure(lambda result, kernel_variance, n_targets=1: _prior_variance_valid(result, kernel_variance, n_targets), "prior variance must preserve the kernel variance or repeat it across targets")
def gp_regression_prior_variance(
    kernel_variance: NDArray[np.float64],
    *,
    n_targets: int = 1,
) -> NDArray[np.float64]:
    """Format unfitted Gaussian-process prior variance outputs from a supplied kernel diagonal."""
    variance = np.asarray(kernel_variance, dtype=np.float64)
    if int(n_targets) == 1:
        return variance
    return np.asarray(
        np.repeat(np.expand_dims(variance, -1), repeats=int(n_targets), axis=-1),
        dtype=np.float64,
    )


@register_atom(witness_gp_regression_prior_std)
@icontract.require(lambda prior_variance: _nonnegative_vector_or_matrix(prior_variance), "prior_variance must be a finite nonempty nonnegative vector or matrix")
@icontract.ensure(lambda result, prior_variance: _prior_std_valid(result, prior_variance), "prior standard deviation must be the elementwise square root of the supplied prior variance")
def gp_regression_prior_std(
    prior_variance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert unfitted Gaussian-process prior variance outputs to standard deviations."""
    return np.sqrt(np.asarray(prior_variance, dtype=np.float64))
