"""Ghost witnesses for Gaussian-process regression prior-prediction atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_positive_int(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be a vector")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def _check_square_matrix(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be a matrix")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    if rows != cols:
        raise ValueError(f"{name} must be square")
    return rows


def witness_gp_regression_prior_target_count(
    *,
    n_targets: int | None = None,
) -> int:
    """Describe resolving the configured prior output count."""
    if n_targets is not None:
        _check_positive_int(n_targets, "n_targets")
        return n_targets
    return 1


def witness_gp_regression_prior_mean(
    n_samples: int,
    *,
    n_targets: int = 1,
) -> AbstractArray:
    """Describe the unfitted prior mean array shape."""
    sample_count = _check_positive_int(n_samples, "n_samples")
    target_count = _check_positive_int(n_targets, "n_targets")
    if target_count == 1:
        return AbstractArray(shape=(sample_count,), dtype="float64")
    return AbstractArray(shape=(sample_count, target_count), dtype="float64")


def witness_gp_regression_prior_covariance(
    kernel_covariance: AbstractArray,
    *,
    n_targets: int = 1,
) -> AbstractArray:
    """Describe the unfitted prior covariance shape."""
    size = _check_square_matrix(kernel_covariance, "kernel_covariance")
    target_count = _check_positive_int(n_targets, "n_targets")
    if target_count == 1:
        return AbstractArray(shape=(size, size), dtype="float64")
    return AbstractArray(shape=(size, size, target_count), dtype="float64")


def witness_gp_regression_prior_variance(
    kernel_variance: AbstractArray,
    *,
    n_targets: int = 1,
) -> AbstractArray:
    """Describe the unfitted prior variance shape."""
    size = _check_vector(kernel_variance, "kernel_variance")
    target_count = _check_positive_int(n_targets, "n_targets")
    if target_count == 1:
        return AbstractArray(shape=(size,), dtype="float64")
    return AbstractArray(shape=(size, target_count), dtype="float64")


def witness_gp_regression_prior_std(
    prior_variance: AbstractArray,
) -> AbstractArray:
    """Describe the standard-deviation output shape from prior variance."""
    if len(prior_variance.shape) == 1:
        size = _check_vector(prior_variance, "prior_variance")
        return AbstractArray(shape=(size,), dtype="float64")
    if len(prior_variance.shape) == 2:
        rows, cols = int(prior_variance.shape[0]), int(prior_variance.shape[1])
        if rows < 1 or cols < 1:
            raise ValueError("prior_variance must be nonempty")
        return AbstractArray(shape=(rows, cols), dtype="float64")
    raise ValueError("prior_variance must be a vector or matrix")
