"""Ghost witnesses for t-SNE scheduling and bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be a vector")
    length = int(values.shape[0])
    if length < 1:
        raise ValueError(f"{name} must be nonempty")
    return length


def witness_tsne_gradient_descent_buffers(
    p0: AbstractArray,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe flattened parameter, update, and gain buffers."""
    length = _check_vector(p0, "p0")
    return (
        AbstractArray(shape=(length,), dtype="float64"),
        AbstractArray(shape=(length,), dtype="float64"),
        AbstractArray(shape=(length,), dtype="float64"),
    )


def witness_tsne_gradient_descent_compute_error(
    iteration: int,
    max_iter: int,
    *,
    n_iter_check: int = 1,
) -> bool:
    """Describe the t-SNE objective-error scheduling predicate."""
    del n_iter_check
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if iteration >= max_iter:
        raise ValueError("iteration must be smaller than max_iter")
    return True


def witness_tsne_gradient_descent_convergence(
    error: float,
    grad: AbstractArray,
    iteration: int,
    best_error: float,
    best_iter: int,
    *,
    n_iter_without_progress: int = 300,
    min_grad_norm: float = 1e-7,
) -> tuple[float, int, float, bool]:
    """Describe best-error tracking and stop checks for one iteration."""
    del error, best_error, n_iter_without_progress, min_grad_norm
    _check_vector(grad, "grad")
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    if best_iter < 0 or best_iter > iteration:
        raise ValueError("best_iter must be between 0 and iteration")
    return 0.0, best_iter, 0.0, False


def witness_tsne_early_exaggeration_scale(probabilities: AbstractArray, early_exaggeration: float) -> AbstractArray:
    """Describe early-exaggeration scaling for dense probability storage."""
    del early_exaggeration
    if len(probabilities.shape) not in {1, 2}:
        raise ValueError("probabilities must be 1D or 2D")
    if any(int(dim) < 1 for dim in probabilities.shape):
        raise ValueError("probabilities must be nonempty")
    return AbstractArray(shape=probabilities.shape, dtype="float64")


def witness_tsne_early_exaggeration_unscale(probabilities: AbstractArray, early_exaggeration: float) -> AbstractArray:
    """Describe early-exaggeration removal for dense probability storage."""
    return witness_tsne_early_exaggeration_scale(probabilities, early_exaggeration)


def witness_tsne_stage_two_required(iteration: int, exploration_max_iter: int, max_iter: int) -> bool:
    """Describe whether sklearn enters the second t-SNE optimization stage."""
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    if exploration_max_iter < 1:
        raise ValueError("exploration_max_iter must be positive")
    if max_iter < exploration_max_iter:
        raise ValueError("max_iter must be at least exploration_max_iter")
    return True
