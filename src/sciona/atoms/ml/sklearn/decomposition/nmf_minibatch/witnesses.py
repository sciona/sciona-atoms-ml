"""Ghost witnesses for MiniBatchNMF scheduling and convergence atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_nmf_minibatch_batch_size(batch_size: int, n_samples: int) -> int:
    """Describe the minibatch size after clamping to the sample count."""
    if batch_size < 1 or n_samples < 1:
        raise ValueError("batch_size and n_samples must be positive")
    return 1


def witness_nmf_minibatch_rho(forget_factor: float, batch_size: int, n_samples: int) -> float:
    """Describe the forgetting-rate power used by MiniBatchNMF."""
    if not 0.0 <= forget_factor <= 1.0:
        raise ValueError("forget_factor must lie in [0, 1]")
    if batch_size < 1 or n_samples < 1:
        raise ValueError("batch_size and n_samples must be positive")
    return 0.0


def witness_nmf_minibatch_mm_gamma(beta_loss: float) -> float:
    """Describe the MM gamma chosen from beta loss."""
    del beta_loss
    return 1.0


def witness_nmf_minibatch_transform_max_iter(max_iter: int, transform_max_iter: int | None) -> int:
    """Describe the transform iteration cap after sklearn defaulting."""
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if transform_max_iter is not None and transform_max_iter < 1:
        raise ValueError("transform_max_iter must be positive when provided")
    return 1


def witness_nmf_minibatch_ewa_cost(
    previous_ewa_cost: float | None,
    batch_cost: float,
    batch_size: int,
    n_samples: int,
) -> float:
    """Describe the updated exponentially weighted average cost."""
    del previous_ewa_cost, batch_cost
    if batch_size < 1 or n_samples < 1:
        raise ValueError("batch_size and n_samples must be positive")
    return 0.0


def witness_nmf_minibatch_h_change_converged(
    H: AbstractArray,
    H_buffer: AbstractArray,
    tol: float,
) -> bool:
    """Describe the H-change stopping predicate."""
    shape = _check_matrix(H, "H")
    if _check_matrix(H_buffer, "H_buffer") != shape:
        raise ValueError("H and H_buffer must have matching shapes")
    if tol < 0.0:
        raise ValueError("tol must be nonnegative")
    return False


def witness_nmf_minibatch_improvement_state(
    ewa_cost: float,
    ewa_cost_min: float | None,
    no_improvement: int,
    max_no_improvement: int | None,
) -> tuple[float, int, bool]:
    """Describe the smoothed-cost improvement tracker update."""
    del ewa_cost, ewa_cost_min, no_improvement, max_no_improvement
    return 0.0, 0, False
