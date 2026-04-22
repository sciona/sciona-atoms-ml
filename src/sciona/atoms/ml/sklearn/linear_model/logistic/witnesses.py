"""Ghost witnesses for sklearn binary logistic-regression objective helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def witness_binary_logistic_positive_probability(raw_prediction: AbstractArray) -> AbstractArray:
    """Describe positive-class probabilities from binary logistic raw scores."""
    n_samples = _check_vector(raw_prediction, "raw_prediction")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_binary_logistic_half_loss_gradient(
    y: AbstractArray,
    raw_prediction: AbstractArray,
    *,
    sample_weight: AbstractArray | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe pointwise binary logistic half loss and raw gradients."""
    n_samples = _check_vector(y, "y")
    if _check_vector(raw_prediction, "raw_prediction") != n_samples:
        raise ValueError("raw_prediction must match y")
    if sample_weight is not None and _check_vector(sample_weight, "sample_weight") != n_samples:
        raise ValueError("sample_weight must match y")
    return (
        AbstractArray(shape=(n_samples,), dtype="float64"),
        AbstractArray(shape=(n_samples,), dtype="float64"),
    )


def witness_binary_logistic_dense_loss_gradient(
    params: AbstractArray,
    X: AbstractArray,
    y: AbstractArray,
    *,
    alpha: float = 0.0,
    sample_weight: AbstractArray | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe dense binary logistic objective value and parameter gradient."""
    del alpha
    n_params = _check_vector(params, "params")
    n_samples, n_features = _check_matrix(X, "X")
    if _check_vector(y, "y") != n_samples:
        raise ValueError("y must match X samples")
    if n_params not in {n_features, n_features + 1}:
        raise ValueError("params must contain coefficients and optional intercept")
    if sample_weight is not None and _check_vector(sample_weight, "sample_weight") != n_samples:
        raise ValueError("sample_weight must match X samples")
    return (
        AbstractArray(shape=(), dtype="float64"),
        AbstractArray(shape=(n_params,), dtype="float64"),
    )
