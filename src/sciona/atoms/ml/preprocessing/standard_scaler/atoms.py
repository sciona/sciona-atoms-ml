"""Standard Scaler preprocessing atoms.

Provides stateless, contracted execution units that compute feature-wise mean
and variance, and apply standard scaling normalization to matrices using
scikit-learn under the hood.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_mean_and_variance,
    witness_apply_z_scaling,
)


@register_atom(witness_compute_mean_and_variance)
@icontract.require(
    lambda matrix: matrix.ndim == 2,
    "matrix must be a 2D array",
)
@icontract.require(
    lambda matrix: np.all(np.isfinite(matrix)),
    "matrix must contain only finite values",
)
@icontract.ensure(
    lambda result, matrix: len(result[0]) == matrix.shape[1],
    "means length must match matrix columns",
)
@icontract.ensure(
    lambda result, matrix: len(result[1]) == matrix.shape[1],
    "variances length must match matrix columns",
)
def compute_mean_and_variance(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute mean and variance vectors for a 2D matrix along axis 0.

    This atom fits a scikit-learn StandardScaler to the input matrix and
    extracts the computed mean and variance attributes per column.

    Args:
        matrix: 2D input matrix of shape (n_samples, n_features).

    Returns:
        A tuple containing (means, variances) where both are 1D arrays of shape (n_features,).
    """
    scaler = StandardScaler()
    scaler.fit(matrix)
    # StandardScaler.mean_ and StandardScaler.var_ are guaranteed to be populated after fit
    assert scaler.mean_ is not None
    assert scaler.var_ is not None
    return scaler.mean_, scaler.var_


@register_atom(witness_apply_z_scaling)
@icontract.require(
    lambda matrix, means: matrix.shape[1] == len(means),
    "number of columns in matrix must match means length",
)
@icontract.require(
    lambda means, variances: len(means) == len(variances),
    "means and variances must have the same length",
)
@icontract.require(
    lambda epsilon: epsilon > 0.0,
    "epsilon must be positive",
)
@icontract.ensure(
    lambda result, matrix: result.shape == matrix.shape,
    "scaled matrix shape must match input matrix shape",
)
@icontract.ensure(
    lambda result: np.all(np.isfinite(result)),
    "scaled matrix must contain only finite values",
)
def apply_z_scaling(
    matrix: NDArray[np.float64],
    means: NDArray[np.float64],
    variances: NDArray[np.float64],
    epsilon: float,
) -> NDArray[np.float64]:
    """Normalize input matrix using pre-computed means and variances.

    Uses a pre-configured scikit-learn StandardScaler instance to perform the
    transformation. Zero-variance columns are handled safely using the epsilon
    threshold to avoid division by zero or tiny values.

    Args:
        matrix: 2D input matrix of shape (n_samples, n_features).
        means: 1D mean vector of shape (n_features,).
        variances: 1D variance vector of shape (n_features,).
        epsilon: Minimum variance threshold to prevent divide-by-zero.

    Returns:
        Scaled matrix of shape (n_samples, n_features).
    """
    scaler = StandardScaler()
    scaler.mean_ = means
    scaler.var_ = variances

    # Handle zero-variance check safely to prevent division by zero.
    scale = np.sqrt(variances)
    scale[variances < epsilon] = 1.0
    scaler.scale_ = scale
    scaler.n_samples_seen_ = len(matrix)

    scaled_matrix = scaler.transform(matrix)
    return scaled_matrix
