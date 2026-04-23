"""Ghost witnesses for Gaussian-process regression linear-algebra atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_square(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows != cols or rows < 1:
        raise ValueError(f"{name} must be nonempty and square")
    return rows


def _check_vector_or_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) == 1:
        rows = int(values.shape[0])
        if rows < 1:
            raise ValueError(f"{name} must be nonempty")
        return rows, 1
    if len(values.shape) == 2:
        rows, cols = int(values.shape[0]), int(values.shape[1])
        if rows < 1 or cols < 1:
            raise ValueError(f"{name} must be nonempty")
        return rows, cols
    raise ValueError(f"{name} must be a vector or matrix")


def witness_gp_regularized_train_kernel(K: AbstractArray, alpha: float | AbstractArray) -> AbstractArray:
    """Describe adding alpha noise to a GP training-kernel diagonal."""
    del alpha
    n_samples = _check_square(K, "K")
    return AbstractArray(shape=(n_samples, n_samples), dtype="float64")


def witness_gp_train_cholesky(K_regularized: AbstractArray) -> AbstractArray:
    """Describe a lower Cholesky factor of a regularized GP kernel."""
    n_samples = _check_square(K_regularized, "K_regularized")
    return AbstractArray(shape=(n_samples, n_samples), dtype="float64")


def witness_gp_dual_coefficients(L: AbstractArray, y_train: AbstractArray) -> AbstractArray:
    """Describe solving GP dual coefficients for training targets."""
    n_samples = _check_square(L, "L")
    target_rows, target_cols = _check_vector_or_matrix(y_train, "y_train")
    if target_rows != n_samples:
        raise ValueError("target rows must match the Cholesky sample count")
    if len(y_train.shape) == 1:
        return AbstractArray(shape=(n_samples,), dtype="float64")
    return AbstractArray(shape=(n_samples, target_cols), dtype="float64")


def witness_gp_log_marginal_likelihood(
    y_train: AbstractArray,
    dual_coefficients: AbstractArray,
    L: AbstractArray,
) -> float:
    """Describe the scalar GP log-marginal likelihood computation."""
    n_samples = _check_square(L, "L")
    if _check_vector_or_matrix(y_train, "y_train") != _check_vector_or_matrix(dual_coefficients, "dual_coefficients"):
        raise ValueError("targets and dual coefficients must have the same shape")
    if int(y_train.shape[0]) != n_samples:
        raise ValueError("target rows must match the Cholesky sample count")
    return 0.0


def witness_gp_posterior_predictive_mean(
    K_trans: AbstractArray,
    dual_coefficients: AbstractArray,
    y_train_mean: float | AbstractArray = 0.0,
    y_train_std: float | AbstractArray = 1.0,
) -> AbstractArray:
    """Describe GP posterior mean shape from cross-kernel and dual weights."""
    del y_train_mean, y_train_std
    if len(K_trans.shape) != 2:
        raise ValueError("K_trans must be 2D")
    n_test, n_train = int(K_trans.shape[0]), int(K_trans.shape[1])
    alpha_rows, alpha_cols = _check_vector_or_matrix(dual_coefficients, "dual_coefficients")
    if alpha_rows != n_train:
        raise ValueError("cross-kernel columns must match dual coefficient rows")
    if len(dual_coefficients.shape) == 1 or alpha_cols == 1:
        return AbstractArray(shape=(n_test,), dtype="float64")
    return AbstractArray(shape=(n_test, alpha_cols), dtype="float64")


def witness_gp_posterior_cross_solve(L: AbstractArray, K_trans: AbstractArray) -> AbstractArray:
    """Describe the triangular posterior cross solve."""
    n_train = _check_square(L, "L")
    if len(K_trans.shape) != 2:
        raise ValueError("K_trans must be 2D")
    n_test, cross_train = int(K_trans.shape[0]), int(K_trans.shape[1])
    if cross_train != n_train:
        raise ValueError("cross-kernel columns must match training count")
    return AbstractArray(shape=(n_train, n_test), dtype="float64")


def witness_gp_posterior_predictive_covariance(
    K_test: AbstractArray,
    V: AbstractArray,
    y_train_std: float | AbstractArray = 1.0,
) -> AbstractArray:
    """Describe GP posterior covariance shape."""
    del y_train_std
    n_test = _check_square(K_test, "K_test")
    if len(V.shape) != 2 or int(V.shape[1]) != n_test:
        raise ValueError("V must have train-by-test shape")
    return AbstractArray(shape=(n_test, n_test), dtype="float64")


def witness_gp_posterior_predictive_std(
    kernel_diag: AbstractArray,
    V: AbstractArray,
    y_train_std: float | AbstractArray = 1.0,
) -> AbstractArray:
    """Describe GP posterior standard-deviation shape."""
    del y_train_std
    if len(kernel_diag.shape) != 1:
        raise ValueError("kernel_diag must be a vector")
    n_test = int(kernel_diag.shape[0])
    if n_test < 1:
        raise ValueError("kernel_diag must be nonempty")
    if len(V.shape) != 2 or int(V.shape[1]) != n_test:
        raise ValueError("V must have train-by-test shape")
    return AbstractArray(shape=(n_test,), dtype="float64")
