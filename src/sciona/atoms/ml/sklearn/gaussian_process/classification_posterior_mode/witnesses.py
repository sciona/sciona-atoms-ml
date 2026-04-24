"""Ghost witnesses for Gaussian-process classification posterior-mode atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    length = int(values.shape[0])
    if length < 1:
        raise ValueError(f"{name} must be nonempty")
    return length


def _check_square(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be two-dimensional")
    rows = int(values.shape[0])
    cols = int(values.shape[1])
    if rows < 1 or cols < 1 or rows != cols:
        raise ValueError(f"{name} must be a nonempty square matrix")
    return rows


def witness_gp_classifier_posterior_mode_initial_latent(
    y_train: AbstractArray,
    *,
    warm_start: bool = False,
    cached_f: AbstractArray | None = None,
) -> AbstractArray:
    """Describe the initial latent vector used for binary Laplace GPC."""
    del warm_start
    n_samples = _check_vector(y_train, "y_train")
    if cached_f is not None:
        if _check_vector(cached_f, "cached_f") != n_samples:
            raise ValueError("cached_f must match y_train length")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_gp_classifier_posterior_mode_converged(
    previous_log_marginal_likelihood: float,
    current_log_marginal_likelihood: float,
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Describe the posterior-mode improvement stopping predicate."""
    del previous_log_marginal_likelihood, current_log_marginal_likelihood, tolerance
    return False


def witness_gp_classifier_posterior_mode(
    K: AbstractArray,
    y_train: AbstractArray,
    *,
    max_iter_predict: int = 100,
    warm_start: bool = False,
    cached_f: AbstractArray | None = None,
) -> tuple[float, AbstractArray, AbstractArray, AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    """Describe the fixed-kernel posterior-mode solve outputs for binary Laplace GPC."""
    del max_iter_predict, warm_start
    n_samples = _check_square(K, "K")
    if _check_vector(y_train, "y_train") != n_samples:
        raise ValueError("y_train must match K")
    if cached_f is not None and _check_vector(cached_f, "cached_f") != n_samples:
        raise ValueError("cached_f must match y_train length")
    return (
        0.0,
        AbstractArray(shape=(n_samples,), dtype="float64"),
        AbstractArray(shape=(n_samples,), dtype="float64"),
        AbstractArray(shape=(n_samples,), dtype="float64"),
        AbstractArray(shape=(n_samples, n_samples), dtype="float64"),
        AbstractArray(shape=(n_samples,), dtype="float64"),
        AbstractArray(shape=(n_samples,), dtype="float64"),
    )
