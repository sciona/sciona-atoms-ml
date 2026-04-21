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


def witness_rational_quadratic_kernel(
    X: AbstractArray,
    Y: AbstractArray | None = None,
    *,
    length_scale: float = 1.0,
    alpha: float = 1.0,
) -> AbstractArray:
    """Describe a rational quadratic covariance matrix."""
    _positive_scalar(length_scale, "length_scale")
    _positive_scalar(alpha, "alpha")
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_rational_quadratic_kernel_diag(X: AbstractArray) -> AbstractArray:
    """Describe the diagonal of a rational quadratic covariance matrix."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_matern_kernel_matrix(
    X: AbstractArray,
    Y: AbstractArray | None = None,
    *,
    length_scale: float | tuple[float, ...] = 1.0,
    nu: float = 1.5,
) -> AbstractArray:
    """Describe a Matern covariance matrix."""
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    rows, cols = _check_xy(X, Y)
    if isinstance(length_scale, tuple) and len(length_scale) not in {1, int(X.shape[1])}:
        raise ValueError("length_scale must be scalar or match feature count")
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_matern_kernel_diag(X: AbstractArray) -> AbstractArray:
    """Describe the diagonal of a Matern covariance matrix."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_exp_sine_squared_kernel(
    X: AbstractArray,
    Y: AbstractArray | None = None,
    *,
    length_scale: float = 1.0,
    periodicity: float = 1.0,
) -> AbstractArray:
    """Describe a periodic covariance matrix."""
    _positive_scalar(length_scale, "length_scale")
    _positive_scalar(periodicity, "periodicity")
    rows, cols = _check_xy(X, Y)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_exp_sine_squared_kernel_diag(X: AbstractArray) -> AbstractArray:
    """Describe the diagonal of a periodic covariance matrix."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def _check_same_matrix_shape(K1: AbstractArray, K2: AbstractArray) -> tuple[int, int]:
    if len(K1.shape) != 2 or len(K2.shape) != 2:
        raise ValueError("kernel matrices must be 2D")
    if K1.shape != K2.shape:
        raise ValueError("kernel matrices must have matching shapes")
    return int(K1.shape[0]), int(K1.shape[1])


def _check_same_diag_shape(d1: AbstractArray, d2: AbstractArray) -> int:
    if len(d1.shape) != 1 or len(d2.shape) != 1:
        raise ValueError("kernel diagonals must be 1D")
    if d1.shape != d2.shape:
        raise ValueError("kernel diagonals must have matching shapes")
    return int(d1.shape[0])


def witness_sum_kernel_matrix(K1: AbstractArray, K2: AbstractArray) -> AbstractArray:
    """Describe elementwise addition of two kernel matrices."""
    rows, cols = _check_same_matrix_shape(K1, K2)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_sum_kernel_diag(d1: AbstractArray, d2: AbstractArray) -> AbstractArray:
    """Describe elementwise addition of two kernel diagonals."""
    rows = _check_same_diag_shape(d1, d2)
    return AbstractArray(shape=(rows,), dtype="float64")


def witness_product_kernel_matrix(K1: AbstractArray, K2: AbstractArray) -> AbstractArray:
    """Describe elementwise multiplication of two kernel matrices."""
    rows, cols = _check_same_matrix_shape(K1, K2)
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_product_kernel_diag(d1: AbstractArray, d2: AbstractArray) -> AbstractArray:
    """Describe elementwise multiplication of two kernel diagonals."""
    rows = _check_same_diag_shape(d1, d2)
    return AbstractArray(shape=(rows,), dtype="float64")


def witness_exponentiation_kernel_matrix(K: AbstractArray, *, exponent: float = 1.0) -> AbstractArray:
    """Describe a kernel matrix raised elementwise to a scalar power."""
    if len(K.shape) != 2:
        raise ValueError("kernel matrix must be 2D")
    if exponent < 0.0:
        raise ValueError("exponent must be non-negative")
    return AbstractArray(shape=(int(K.shape[0]), int(K.shape[1])), dtype="float64")


def witness_exponentiation_kernel_diag(d: AbstractArray, *, exponent: float = 1.0) -> AbstractArray:
    """Describe a kernel diagonal raised elementwise to a scalar power."""
    if len(d.shape) != 1:
        raise ValueError("kernel diagonal must be 1D")
    if exponent < 0.0:
        raise ValueError("exponent must be non-negative")
    return AbstractArray(shape=(int(d.shape[0]),), dtype="float64")


def witness_compound_kernel_stack(kernels: tuple[AbstractArray, ...]) -> AbstractArray:
    """Describe stacking multiple kernel matrices along a final axis."""
    if not kernels:
        raise ValueError("at least one kernel matrix is required")
    first_shape = kernels[0].shape
    if len(first_shape) != 2:
        raise ValueError("kernel matrices must be 2D")
    for kernel in kernels[1:]:
        if len(kernel.shape) != 2 or kernel.shape != first_shape:
            raise ValueError("kernel matrices must have matching shapes")
    return AbstractArray(shape=(int(first_shape[0]), int(first_shape[1]), len(kernels)), dtype="float64")


def witness_compound_kernel_diag_stack(diagonals: tuple[AbstractArray, ...]) -> AbstractArray:
    """Describe stacking multiple kernel diagonals into columns."""
    if not diagonals:
        raise ValueError("at least one kernel diagonal is required")
    first_shape = diagonals[0].shape
    if len(first_shape) != 1:
        raise ValueError("kernel diagonals must be 1D")
    for diagonal in diagonals[1:]:
        if len(diagonal.shape) != 1 or diagonal.shape != first_shape:
            raise ValueError("kernel diagonals must have matching shapes")
    return AbstractArray(shape=(int(first_shape[0]), len(diagonals)), dtype="float64")
