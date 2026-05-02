"""Ghost witnesses for Gaussian-process regression sample_y shell atoms."""

from __future__ import annotations

from numpy.typing import NDArray

from sciona.ghost.abstract import AbstractArray


def witness_gp_sample_y_use_multioutput_branch(
    y_mean: NDArray[float],
) -> bool:
    """Describe whether GaussianProcessRegressor.sample_y uses the multi-output branch."""
    del y_mean
    return False


def witness_gp_sample_y_result(
    y_mean: AbstractArray,
    y_cov: AbstractArray,
    *,
    n_samples: int = 1,
    random_state: int | None = 0,
) -> AbstractArray:
    """Describe the sample_y result shape from supplied predictive means and covariances."""
    del y_cov
    del random_state
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if len(y_mean.shape) == 1:
        return AbstractArray(shape=(int(y_mean.shape[0]), n_samples), dtype="float64")
    if len(y_mean.shape) == 2:
        return AbstractArray(shape=(int(y_mean.shape[0]), int(y_mean.shape[1]), n_samples), dtype="float64")
    raise ValueError("y_mean must be 1D or 2D")
