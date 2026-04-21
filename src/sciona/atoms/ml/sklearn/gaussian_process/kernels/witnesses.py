"""Ghost witnesses for sklearn Gaussian process kernel atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_xy(X: AbstractArray, Y: AbstractArray | None) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if Y is None:
        return int(X.shape[0]), int(X.shape[0])
    if len(Y.shape) != 2:
        raise ValueError("Y must be 2D")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have matching feature counts")
    return int(X.shape[0]), int(Y.shape[0])


def _positive_scalar(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def witness_constant_kernel(X: AbstractArray, Y: AbstractArray | None = None, *, constant_value: float = 1.0) -> AbstractArray:
    """Describe a constant covariance matrix."""
    _positive_scalar(constant_value, "constant_value")
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_constant_kernel_diag(X: AbstractArray, *, constant_value: float = 1.0) -> AbstractArray:
    """Describe the diagonal of a constant covariance matrix."""
    _positive_scalar(constant_value, "constant_value")
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_white_kernel(X: AbstractArray, Y: AbstractArray | None = None, *, noise_level: float = 1.0) -> AbstractArray:
    """Describe a white-noise covariance matrix."""
    _positive_scalar(noise_level, "noise_level")
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_white_kernel_diag(X: AbstractArray, *, noise_level: float = 1.0) -> AbstractArray:
    """Describe the diagonal of a white-noise covariance matrix."""
    _positive_scalar(noise_level, "noise_level")
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_dot_product_kernel(X: AbstractArray, Y: AbstractArray | None = None, *, sigma_0: float = 1.0) -> AbstractArray:
    """Describe an offset dot-product covariance matrix."""
    if sigma_0 < 0.0:
        raise ValueError("sigma_0 must be non-negative")
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_dot_product_kernel_diag(X: AbstractArray, *, sigma_0: float = 1.0) -> AbstractArray:
    """Describe the diagonal of an offset dot-product covariance matrix."""
    if sigma_0 < 0.0:
        raise ValueError("sigma_0 must be non-negative")
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_rbf_kernel_matrix(
    X: AbstractArray,
    Y: AbstractArray | None = None,
    *,
    length_scale: float | tuple[float, ...] = 1.0,
) -> AbstractArray:
    """Describe a squared-exponential covariance matrix."""
    rows, cols = _check_xy(X, Y)
    if isinstance(length_scale, tuple) and len(length_scale) not in {1, int(X.shape[1])}:
        raise ValueError("length_scale must be scalar or match feature count")
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_rbf_kernel_diag(X: AbstractArray) -> AbstractArray:
    """Describe the diagonal of a squared-exponential covariance matrix."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")
