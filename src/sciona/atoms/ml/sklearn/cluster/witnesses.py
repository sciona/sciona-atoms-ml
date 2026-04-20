"""Ghost witnesses for selected sklearn cluster atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import AffinityPropagationState


def witness_affinity_propagation(
    S: AbstractArray,
    *,
    preference: float | AbstractArray | None = None,
    convergence_iter: int = 15,
    max_iter: int = 200,
    damping: float = 0.5,
    copy: bool = True,
    verbose: bool = False,
    return_n_iter: bool = False,
    random_state: int | None = None,
) -> tuple[AbstractArray, AbstractArray] | tuple[AbstractArray, AbstractArray, int]:
    """Describe cluster centers and labels from a similarity matrix."""
    del preference, copy, verbose, random_state
    n_samples = _check_square(S)
    _check_iteration_parameters(convergence_iter, max_iter, damping)
    centers = AbstractArray(shape=(n_samples,), dtype="int64", min_val=0)
    labels = AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)
    if return_n_iter:
        return centers, labels, max_iter
    return centers, labels


def witness_affinity_propagation_fit(
    X: AbstractArray,
    *,
    damping: float = 0.5,
    max_iter: int = 200,
    convergence_iter: int = 15,
    copy: bool = True,
    preference: float | AbstractArray | None = None,
    affinity: str = "euclidean",
    verbose: bool = False,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe fitting affinity propagation into immutable state."""
    del copy, preference, verbose, random_state
    n_samples, _ = _check_2d(X)
    if affinity not in {"euclidean", "precomputed"}:
        raise ValueError("affinity must be 'euclidean' or 'precomputed'")
    if affinity == "precomputed" and X.shape[0] != X.shape[1]:
        raise ValueError("precomputed affinity must be square")
    _check_iteration_parameters(convergence_iter, max_iter, damping)
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)


def witness_affinity_propagation_predict(
    X: AbstractArray,
    state: AffinityPropagationState,
) -> AbstractArray:
    """Describe nearest-center prediction from fitted affinity propagation state."""
    n_samples, n_features = _check_2d(X)
    if state.affinity == "precomputed":
        raise ValueError("predict is not supported for precomputed affinity")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1)


def _check_2d(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return int(X.shape[0]), int(X.shape[1])


def _check_square(S: AbstractArray) -> int:
    rows, cols = _check_2d(S)
    if rows != cols:
        raise ValueError("similarity matrix must be square")
    return rows


def _check_iteration_parameters(convergence_iter: int, max_iter: int, damping: float) -> None:
    if convergence_iter < 1:
        raise ValueError("convergence_iter must be at least one")
    if max_iter < 1:
        raise ValueError("max_iter must be at least one")
    if not 0.5 <= damping < 1.0:
        raise ValueError("damping must be in [0.5, 1.0)")
