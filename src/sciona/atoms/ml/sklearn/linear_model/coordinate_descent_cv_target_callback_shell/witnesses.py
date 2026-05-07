"""Ghost witnesses for sklearn coordinate-descent CV target callback-shell atoms."""

from __future__ import annotations


def witness_cd_cv_is_multitask_result(multitask: object) -> object:
    """Describe the boolean result from self._is_multitask()."""
    return multitask


def witness_cd_cv_column_or_1d_args(y: object, multitask: object) -> object:
    """Describe positional args for column_or_1d(y, warn=True)."""
    return y, multitask


def witness_cd_cv_column_or_1d_result(normalized_y: object) -> object:
    """Describe y returned by deferred column_or_1d(...)."""
    return normalized_y


def witness_cd_cv_check_sample_weight_args(sample_weight: object, X: object) -> object:
    """Describe positional args for _check_sample_weight(sample_weight, X, ...)."""
    return sample_weight, X


def witness_cd_cv_check_sample_weight_kwargs(x_dtype: object) -> object:
    """Describe dtype kwargs for _check_sample_weight(..., dtype=X.dtype)."""
    return x_dtype


def witness_cd_cv_checked_sample_weight(checked_sample_weight: object) -> object:
    """Describe sample_weight returned by deferred _check_sample_weight(...)."""
    return checked_sample_weight
