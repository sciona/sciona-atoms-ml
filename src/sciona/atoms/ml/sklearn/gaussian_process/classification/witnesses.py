"""Ghost witnesses for Gaussian-process classification Laplace atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def _check_square(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows != cols or rows < 1:
        raise ValueError(f"{name} must be nonempty and square")
    return rows


def witness_gp_classifier_laplace_newton_step(
    K: AbstractArray,
    y_train: AbstractArray,
    f: AbstractArray,
) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    """Describe one binary Laplace Newton step."""
    n_samples = _check_square(K, "K")
    if _check_vector(y_train, "y_train") != n_samples:
        raise ValueError("y_train length must match K")
    if _check_vector(f, "f") != n_samples:
        raise ValueError("f length must match K")
    vector = AbstractArray(shape=(n_samples,), dtype="float64")
    matrix = AbstractArray(shape=(n_samples, n_samples), dtype="float64")
    return vector, vector, vector, matrix, vector, vector


def witness_gp_classifier_laplace_log_marginal_likelihood(
    y_train: AbstractArray,
    f: AbstractArray,
    a: AbstractArray,
    L: AbstractArray,
) -> float:
    """Describe the scalar binary Laplace log-marginal likelihood."""
    n_samples = _check_square(L, "L")
    if _check_vector(y_train, "y_train") != n_samples:
        raise ValueError("y_train length must match L")
    if _check_vector(f, "f") != n_samples:
        raise ValueError("f length must match L")
    if _check_vector(a, "a") != n_samples:
        raise ValueError("a length must match L")
    return 0.0


def witness_gp_classifier_posterior_mean(
    K_star: AbstractArray,
    y_train: AbstractArray,
    pi: AbstractArray,
) -> AbstractArray:
    """Describe binary GP classification posterior means at test points."""
    if len(K_star.shape) != 2:
        raise ValueError("K_star must be 2D")
    n_train, n_test = int(K_star.shape[0]), int(K_star.shape[1])
    if n_train < 1 or n_test < 1:
        raise ValueError("K_star must be nonempty")
    if _check_vector(y_train, "y_train") != n_train:
        raise ValueError("y_train length must match K_star rows")
    if _check_vector(pi, "pi") != n_train:
        raise ValueError("pi length must match K_star rows")
    return AbstractArray(shape=(n_test,), dtype="float64")


def witness_gp_classifier_posterior_cross_solve(
    L: AbstractArray,
    W_sr: AbstractArray,
    K_star: AbstractArray,
) -> AbstractArray:
    """Describe the posterior triangular solve used in binary GP classification."""
    n_train = _check_square(L, "L")
    if _check_vector(W_sr, "W_sr") != n_train:
        raise ValueError("W_sr length must match L")
    if len(K_star.shape) != 2:
        raise ValueError("K_star must be 2D")
    cross_train, n_test = int(K_star.shape[0]), int(K_star.shape[1])
    if cross_train != n_train or n_test < 1:
        raise ValueError("K_star shape must be train-by-test")
    return AbstractArray(shape=(n_train, n_test), dtype="float64")


def witness_gp_classifier_posterior_variance(
    kernel_diag: AbstractArray,
    v: AbstractArray,
) -> AbstractArray:
    """Describe posterior latent variances at test points."""
    n_test = _check_vector(kernel_diag, "kernel_diag")
    if len(v.shape) != 2 or int(v.shape[1]) != n_test:
        raise ValueError("v must be train-by-test")
    return AbstractArray(shape=(n_test,), dtype="float64")


def witness_gp_classifier_predictive_probability(
    f_star: AbstractArray,
    var_f_star: AbstractArray,
) -> AbstractArray:
    """Describe binary predictive probabilities at test points."""
    n_test = _check_vector(f_star, "f_star")
    if _check_vector(var_f_star, "var_f_star") != n_test:
        raise ValueError("var_f_star length must match f_star")
    return AbstractArray(shape=(n_test,), dtype="float64")


def witness_gp_classifier_predictive_proba(pi_star: AbstractArray) -> AbstractArray:
    """Describe two-column class-probability output."""
    n_test = _check_vector(pi_star, "pi_star")
    return AbstractArray(shape=(n_test, 2), dtype="float64")
