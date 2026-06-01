"""Ghost witnesses for standard scaler preprocessing atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_compute_mean_and_variance(
    matrix: AbstractArray,
) -> tuple[AbstractArray, AbstractArray]:
    """Compute mean and variance vectors for a 2D matrix along axis 0."""
    # Input matrix shape is (n_samples, n_features). Output means and variances are shape (n_features,).
    d = matrix.shape[1] if len(matrix.shape) > 1 else 1
    means = AbstractArray(shape=(d,), dtype="float64")
    variances = AbstractArray(shape=(d,), dtype="float64")
    return means, variances


def witness_apply_z_scaling(
    matrix: AbstractArray,
    means: AbstractArray,
    variances: AbstractArray,
    epsilon: float,
) -> AbstractArray:
    """Normalize input matrix using pre-computed means and variances."""
    # Element-wise transformation preserving shape.
    return AbstractArray(shape=matrix.shape, dtype="float64")
