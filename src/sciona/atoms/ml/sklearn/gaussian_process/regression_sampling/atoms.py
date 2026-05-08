"""Gaussian-process regression sampling atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_sample_y_multi_output,
    witness_gp_sample_y_single_output,
)

RandomStateLike = int | np.random.RandomState | None

def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))

def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))

def _finite_tensor3(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 3
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and array.shape[2] >= 1
        and np.all(np.isfinite(array))
    )

def _symmetric_psd(matrix: NDArray[np.float64]) -> bool:
    values = np.asarray(matrix, dtype=np.float64)
    if not (_finite_matrix(values) and values.shape[0] == values.shape[1]):
        return False
    if not np.allclose(values, values.T):
        return False
    return bool(np.all(np.linalg.eigvalsh(values) >= -1e-10))

def _single_output_inputs_valid(y_mean: NDArray[np.float64], y_cov: NDArray[np.float64], n_samples: int) -> bool:
    return bool(
        _finite_vector(y_mean)
        and _symmetric_psd(y_cov)
        and np.asarray(y_cov, dtype=np.float64).shape[0] == np.asarray(y_mean, dtype=np.float64).shape[0]
        and isinstance(n_samples, int)
        and not isinstance(n_samples, bool)
        and n_samples >= 1
    )

def _multi_output_inputs_valid(y_mean: NDArray[np.float64], y_cov: NDArray[np.float64], n_samples: int) -> bool:
    if not (_finite_matrix(y_mean) and _finite_tensor3(y_cov)):
        return False
    mean = np.asarray(y_mean, dtype=np.float64)
    cov = np.asarray(y_cov, dtype=np.float64)
    return bool(
        cov.shape[0] == mean.shape[0]
        and cov.shape[1] == mean.shape[0]
        and cov.shape[2] == mean.shape[1]
        and all(_symmetric_psd(cov[..., target]) for target in range(cov.shape[2]))
        and isinstance(n_samples, int)
        and not isinstance(n_samples, bool)
        and n_samples >= 1
    )

def _single_output_result_valid(result: NDArray[np.float64], y_mean: NDArray[np.float64], n_samples: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (np.asarray(y_mean, dtype=np.float64).shape[0], int(n_samples)) and np.all(np.isfinite(values)))

def _multi_output_result_valid(result: NDArray[np.float64], y_mean: NDArray[np.float64], n_samples: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    mean = np.asarray(y_mean, dtype=np.float64)
    return bool(values.shape == (mean.shape[0], mean.shape[1], int(n_samples)) and np.all(np.isfinite(values)))

@register_atom(witness_gp_sample_y_single_output)
@icontract.require(lambda y_mean, y_cov, n_samples=1: _single_output_inputs_valid(y_mean, y_cov, n_samples), "y_mean and y_cov must describe a finite symmetric positive-semidefinite single-output predictive Gaussian, with n_samples >= 1")
@icontract.ensure(lambda result, y_mean, n_samples=1: _single_output_result_valid(result, y_mean, n_samples), "sampled outputs must have shape (n_points, n_samples)")
def gp_sample_y_single_output(
    y_mean: NDArray[np.float64],
    y_cov: NDArray[np.float64],
    *,
    n_samples: int = 1,
    random_state: RandomStateLike = 0,
) -> NDArray[np.float64]:
    from sklearn.utils import check_random_state
    """Draw single-output Gaussian-process samples from a supplied predictive mean and covariance."""
    rng = check_random_state(random_state)
    return np.asarray(
        rng.multivariate_normal(
            np.asarray(y_mean, dtype=np.float64),
            np.asarray(y_cov, dtype=np.float64),
            int(n_samples),
        ).T,
        dtype=np.float64,
    )

@register_atom(witness_gp_sample_y_multi_output)
@icontract.require(lambda y_mean, y_cov, n_samples=1: _multi_output_inputs_valid(y_mean, y_cov, n_samples), "y_mean and y_cov must describe finite compatible multi-output predictive Gaussians, with n_samples >= 1")
@icontract.ensure(lambda result, y_mean, n_samples=1: _multi_output_result_valid(result, y_mean, n_samples), "sampled outputs must have shape (n_points, n_targets, n_samples)")
def gp_sample_y_multi_output(
    y_mean: NDArray[np.float64],
    y_cov: NDArray[np.float64],
    *,
    n_samples: int = 1,
    random_state: RandomStateLike = 0,
) -> NDArray[np.float64]:
    from sklearn.utils import check_random_state
    """Draw multi-output Gaussian-process samples from supplied per-target predictive means and covariances."""
    rng = check_random_state(random_state)
    mean = np.asarray(y_mean, dtype=np.float64)
    cov = np.asarray(y_cov, dtype=np.float64)
    samples = [
        rng.multivariate_normal(mean[:, target], cov[..., target], int(n_samples)).T[:, np.newaxis]
        for target in range(mean.shape[1])
    ]
    return np.asarray(np.hstack(samples), dtype=np.float64)
