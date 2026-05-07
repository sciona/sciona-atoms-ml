"""Ghost witnesses for sklearn coordinate-descent estimator pre-fit shell atoms."""

from __future__ import annotations


def witness_cd_estimator_should_copy(copy_X: object, x_copied: object) -> object:
    """Describe the should_copy flag before _pre_fit(...)."""
    return copy_X, x_copied


def witness_cd_estimator_pre_fit_args(X: object, y: object, precompute: object) -> object:
    """Describe positional args for _pre_fit(X, y, None, precompute, ...)."""
    return X, y, precompute


def witness_cd_estimator_pre_fit_kwargs(
    fit_intercept: object, should_copy: object, check_input: object, sample_weight: object
) -> object:
    """Describe kwargs for _pre_fit(...)."""
    return fit_intercept, should_copy, check_input, sample_weight


def witness_cd_estimator_set_order_required(
    check_input: object, sample_weight: object
) -> object:
    """Describe whether _set_order(X, y, order='F') runs."""
    return check_input, sample_weight


def witness_cd_estimator_set_order_args(X: object, y: object) -> object:
    """Describe positional args for _set_order(X, y, order='F')."""
    return X, y


def witness_cd_estimator_y_column_vector_required(y_ndim: object) -> object:
    """Describe whether y is expanded with y[:, np.newaxis]."""
    return y_ndim


def witness_cd_estimator_y_column_vector(y: object) -> object:
    """Describe y[:, np.newaxis] normalization."""
    return y


def witness_cd_estimator_xy_column_vector_required(Xy: object) -> object:
    """Describe whether Xy is expanded with Xy[:, np.newaxis]."""
    return Xy


def witness_cd_estimator_xy_column_vector(Xy: object) -> object:
    """Describe Xy[:, np.newaxis] normalization."""
    return Xy


def witness_cd_estimator_n_targets(y_shape: object) -> object:
    """Describe n_targets extracted from y.shape[1]."""
    return y_shape
