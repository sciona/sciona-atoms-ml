"""Ghost witnesses for sklearn NMF helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_nmf_beta_loss_to_float(beta_loss: float | str) -> float:
    """Describe converting a named beta loss to its numeric value."""
    del beta_loss
    return 0.0


def witness_nmf_trace_dot(X: AbstractArray, Y: AbstractArray) -> float:
    """Describe the flattened trace product used by sklearn NMF helpers."""
    shape = _check_matrix(X, "X")
    if _check_matrix(Y, "Y") != shape:
        raise ValueError("X and Y must have the same shape")
    return 0.0


def witness_nmf_beta_divergence(
    X: AbstractArray,
    W: AbstractArray,
    H: AbstractArray,
    beta: float | str,
    *,
    square_root: bool = False,
) -> float:
    """Describe beta-divergence between X and W times H."""
    del beta, square_root
    n_samples, n_features = _check_matrix(X, "X")
    w_samples, n_components = _check_matrix(W, "W")
    h_components, h_features = _check_matrix(H, "H")
    if w_samples != n_samples:
        raise ValueError("W sample count must match X")
    if h_features != n_features:
        raise ValueError("H feature count must match X")
    if h_components != n_components:
        raise ValueError("W and H component counts must match")
    return 0.0
