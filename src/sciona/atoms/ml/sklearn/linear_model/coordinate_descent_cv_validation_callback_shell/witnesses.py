"""Ghost witnesses for sklearn coordinate-descent CV validation callback-shell atoms."""

from __future__ import annotations


def witness_cd_cv_validate_data_args(estimator: object, X: object, y: object) -> object:
    """Describe positional args for validate_data(self, X, y, ...)."""
    return estimator, X, y


def witness_cd_cv_validate_data_kwargs(check_x_params: object, check_y_params: object) -> object:
    """Describe validate_separately kwargs for validate_data(...)."""
    return check_x_params, check_y_params


def witness_cd_cv_validated_x(validated_pair: object) -> object:
    """Describe X unpacking from validate_data(...) output."""
    return validated_pair


def witness_cd_cv_validated_y(validated_pair: object) -> object:
    """Describe y unpacking from validate_data(...) output."""
    return validated_pair


def witness_cd_cv_check_consistent_length_args(X: object, y: object) -> object:
    """Describe positional args for check_consistent_length(X, y)."""
    return X, y
