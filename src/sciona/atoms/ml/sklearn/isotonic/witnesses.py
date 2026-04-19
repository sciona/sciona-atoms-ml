"""Ghost witnesses for sklearn isotonic regression atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import IsotonicRegressionState


def _check_vector(x: AbstractArray, name: str) -> int:
    if len(x.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    return int(x.shape[0])


def _check_input_x(X: AbstractArray) -> int:
    if len(X.shape) == 1:
        return int(X.shape[0])
    if len(X.shape) == 2 and X.shape[1] == 1:
        return int(X.shape[0])
    raise ValueError("X must be 1D or 2D with one feature")


def witness_isotonic_regression(
    y: AbstractArray,
    *,
    sample_weight: AbstractArray | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    increasing: bool = True,
) -> AbstractArray:
    """Describe the monotone fitted response vector."""
    del y_min, y_max, increasing
    n_samples = _check_vector(y, "y")
    if sample_weight is not None and _check_vector(sample_weight, "sample_weight") != n_samples:
        raise ValueError("sample_weight must match y length")
    return AbstractArray(shape=(n_samples,), dtype=y.dtype)


def witness_isotonic_regression_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    sample_weight: AbstractArray | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    increasing: bool | str = True,
    out_of_bounds: str = "nan",
) -> AbstractArray:
    """Describe learned isotonic threshold values."""
    del y_min, y_max, increasing
    n_samples = _check_input_x(X)
    if _check_vector(y, "y") != n_samples:
        raise ValueError("X and y must have equal sample count")
    if sample_weight is not None and _check_vector(sample_weight, "sample_weight") != n_samples:
        raise ValueError("sample_weight must match X length")
    if out_of_bounds not in {"nan", "clip", "raise"}:
        raise ValueError("out_of_bounds must be 'nan', 'clip', or 'raise'")
    return AbstractArray(shape=(n_samples,), dtype=X.dtype)


def witness_isotonic_regression_transform(
    T: AbstractArray,
    state: IsotonicRegressionState,
) -> AbstractArray:
    """Describe interpolated predictions from fitted isotonic state."""
    n_samples = _check_input_x(T)
    if state.x_thresholds.shape != state.y_thresholds.shape:
        raise ValueError("state thresholds must have matching shapes")
    return AbstractArray(shape=(n_samples,), dtype=T.dtype)


def witness_isotonic_regression_predict(
    T: AbstractArray,
    state: IsotonicRegressionState,
) -> AbstractArray:
    """Describe prediction output from fitted isotonic state."""
    return witness_isotonic_regression_transform(T, state)
