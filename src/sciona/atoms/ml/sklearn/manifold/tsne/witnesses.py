"""Ghost witnesses for exact-method t-SNE helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be a vector")
    length = int(values.shape[0])
    if length < 1:
        raise ValueError(f"{name} must be nonempty")
    return length


def _check_square(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows != cols or rows < 2:
        raise ValueError(f"{name} must be square with at least two samples")
    return rows


def witness_tsne_exact_joint_probabilities(conditional_probabilities: AbstractArray) -> AbstractArray:
    """Describe dense t-SNE probability symmetrization and condensation."""
    n_samples = _check_square(conditional_probabilities, "conditional_probabilities")
    return AbstractArray(shape=(n_samples * (n_samples - 1) // 2,), dtype="float64")


def witness_tsne_exact_kl_divergence(
    params: AbstractArray,
    P: AbstractArray,
    degrees_of_freedom: int,
    n_samples: int,
    n_components: int,
    *,
    skip_num_points: int = 0,
    compute_error: bool = True,
) -> tuple[float, AbstractArray]:
    """Describe the exact t-SNE KL objective and parameter gradient."""
    del compute_error
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    if n_samples < 2 or n_components < 1:
        raise ValueError("sample and component counts must be positive")
    if skip_num_points != 0:
        raise ValueError("only full-gradient exact t-SNE is covered")
    if _check_vector(params, "params") != n_samples * n_components:
        raise ValueError("params length must match n_samples times n_components")
    if _check_vector(P, "P") != n_samples * (n_samples - 1) // 2:
        raise ValueError("P length must be the condensed sample-pair count")
    return 0.0, AbstractArray(shape=(n_samples * n_components,), dtype="float64")


def witness_tsne_gradient_descent_update(
    p: AbstractArray,
    update: AbstractArray,
    gains: AbstractArray,
    grad: AbstractArray,
    *,
    momentum: float = 0.8,
    learning_rate: float = 200.0,
    min_gain: float = 0.01,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe one t-SNE adaptive-gain momentum update."""
    del momentum, learning_rate, min_gain
    length = _check_vector(p, "p")
    if _check_vector(update, "update") != length:
        raise ValueError("update length must match p")
    if _check_vector(gains, "gains") != length:
        raise ValueError("gains length must match p")
    if _check_vector(grad, "grad") != length:
        raise ValueError("grad length must match p")
    return (
        AbstractArray(shape=(length,), dtype="float64"),
        AbstractArray(shape=(length,), dtype="float64"),
        AbstractArray(shape=(length,), dtype="float64"),
    )
