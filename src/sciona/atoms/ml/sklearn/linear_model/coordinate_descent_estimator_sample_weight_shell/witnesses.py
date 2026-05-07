"""Ghost witnesses for sklearn coordinate-descent estimator sample-weight shell atoms."""

from __future__ import annotations


def witness_cd_estimator_sample_weight_after_scalar_guard(sample_weight: object) -> object:
    """Describe sample_weight after the scalar-number guard."""
    return sample_weight


def witness_cd_estimator_sample_weight_check_required(
    sample_weight: object, check_input: object
) -> object:
    """Describe whether _check_sample_weight is called."""
    return sample_weight, check_input


def witness_cd_estimator_check_sample_weight_args(sample_weight: object, X: object) -> object:
    """Describe positional args for _check_sample_weight(sample_weight, X, ...)."""
    return sample_weight, X


def witness_cd_estimator_check_sample_weight_kwargs(x_dtype: object) -> object:
    """Describe dtype kwargs for _check_sample_weight(..., dtype=X.dtype)."""
    return x_dtype


def witness_cd_estimator_checked_sample_weight(checked_sample_weight: object) -> object:
    """Describe sample_weight returned by deferred _check_sample_weight(...)."""
    return checked_sample_weight


def witness_cd_estimator_sample_weight_rescale_factor(
    sample_weight: object, n_samples: object
) -> object:
    """Describe the n_samples / sum(sample_weight) rescale factor."""
    return sample_weight, n_samples


def witness_cd_estimator_sample_weight_rescaled(
    sample_weight: object, n_samples: object
) -> object:
    """Describe sample_weight after rescaling to sum to n_samples."""
    return sample_weight, n_samples
