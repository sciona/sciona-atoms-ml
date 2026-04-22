"""Ghost witnesses for sklearn generalized-linear-model objective helpers."""

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


def witness_glm_linear_raw_prediction(
    X: AbstractArray,
    coef: AbstractArray,
    *,
    intercept: float = 0.0,
) -> AbstractArray:
    """Describe a GLM linear prediction from supplied coefficients."""
    del intercept
    n_samples, n_features = _check_matrix(X, "X")
    if _check_vector(coef, "coef") != n_features:
        raise ValueError("coef must match X features")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_glm_log_link_half_loss_gradient(
    y: AbstractArray,
    raw_prediction: AbstractArray,
    *,
    family: str,
    power: float = 1.5,
    sample_weight: AbstractArray | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe pointwise log-link GLM half loss and raw gradients."""
    del family, power
    n_samples = _check_vector(y, "y")
    if _check_vector(raw_prediction, "raw_prediction") != n_samples:
        raise ValueError("raw_prediction must match y")
    if sample_weight is not None and _check_vector(sample_weight, "sample_weight") != n_samples:
        raise ValueError("sample_weight must match y")
    return (
        AbstractArray(shape=(n_samples,), dtype="float64"),
        AbstractArray(shape=(n_samples,), dtype="float64"),
    )


def witness_glm_dense_loss_gradient(
    params: AbstractArray,
    X: AbstractArray,
    y: AbstractArray,
    *,
    family: str,
    alpha: float = 0.0,
    power: float = 1.5,
    sample_weight: AbstractArray | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe dense GLM objective value and parameter gradient."""
    del family, alpha, power
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
