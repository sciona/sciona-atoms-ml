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


def witness_nmf_random_initialize(
    X: AbstractArray,
    n_components: int,
    *,
    random_state: int | object | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe random factor initialization for nonnegative matrix factorization."""
    del random_state
    n_samples, n_features = _check_matrix(X, "X")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    return (
        AbstractArray(shape=(n_samples, n_components), dtype="float64", min_val=0.0),
        AbstractArray(shape=(n_components, n_features), dtype="float64", min_val=0.0),
    )


def witness_nmf_nndsvd_from_svd(
    U: AbstractArray,
    S: AbstractArray,
    V: AbstractArray,
    init: str,
    data_mean: float,
    *,
    eps: float = 1e-6,
    random_state: int | object | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe NNDSVD factor reconstruction from a supplied SVD triplet."""
    del init, data_mean, eps, random_state
    n_samples, n_components = _check_matrix(U, "U")
    if len(S.shape) != 1:
        raise ValueError("S must be one-dimensional")
    if int(S.shape[0]) != n_components:
        raise ValueError("S must match the component axis of U")
    v_components, n_features = _check_matrix(V, "V")
    if v_components != n_components:
        raise ValueError("V must share the component axis with U")
    return (
        AbstractArray(shape=(n_samples, n_components), dtype="float64", min_val=0.0),
        AbstractArray(shape=(n_components, n_features), dtype="float64", min_val=0.0),
    )


def witness_nmf_check_init_matrix(
    A: AbstractArray,
    shape: tuple[int | str, int | str],
    whom: str,
) -> AbstractArray:
    """Describe the validated matrix returned for factorization initialization."""
    del whom
    n_rows, n_cols = _check_matrix(A, "A")
    if len(shape) != 2:
        raise ValueError("shape must have two entries")
    expected_rows, expected_cols = shape
    if expected_rows != "auto" and int(expected_rows) != n_rows:
        raise ValueError("A rows must match the requested shape")
    if expected_cols != "auto" and int(expected_cols) != n_cols:
        raise ValueError("A columns must match the requested shape")
    return AbstractArray(shape=(n_rows, n_cols), dtype="float64", min_val=0.0)
