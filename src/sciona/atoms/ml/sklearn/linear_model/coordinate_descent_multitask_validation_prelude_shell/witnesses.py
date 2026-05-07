"""Ghost witnesses for sklearn multi-output validation prelude atoms."""

from __future__ import annotations


def witness_cd_multitask_check_x_params(copy_X: object, fit_intercept: object) -> object:
    """Describe X-validation parameters."""
    return copy_X, fit_intercept


def witness_cd_multitask_check_y_params(y_context: object) -> object:
    """Describe y-validation parameters."""
    return y_context


def witness_cd_multitask_validate_data_args(
    estimator: object, X: object, y: object
) -> object:
    """Describe validate_data positional arguments."""
    return estimator, X, y


def witness_cd_multitask_validate_data_kwargs(
    check_X_params: object, check_y_params: object
) -> object:
    """Describe validate_data keyword arguments."""
    return check_X_params, check_y_params


def witness_cd_multitask_consistent_length_args(X: object, y: object) -> object:
    """Describe check_consistent_length positional arguments."""
    return X, y


def witness_cd_multitask_y_astype_dtype(y: object, x_dtype: object) -> object:
    """Describe y dtype casting."""
    return y, x_dtype


def witness_cd_multitask_shape_counts(x_shape: object, y_shape: object) -> object:
    """Describe sample, feature, and target count extraction."""
    return x_shape, y_shape
