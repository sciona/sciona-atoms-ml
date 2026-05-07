"""Ghost witnesses for sklearn coordinate-descent CV alpha validation callback shells."""

from __future__ import annotations


def witness_cd_cv_user_alpha_validation_required(alphas_is_none: object) -> object:
    """Describe the user-alpha validation branch predicate."""
    return alphas_is_none


def witness_cd_cv_alpha_check_scalar_kwargs(target_type: object) -> object:
    """Describe fixed kwargs for check_scalar on user-provided alphas."""
    return target_type


def witness_cd_cv_alpha_check_scalar_args(alpha: object, index: object) -> object:
    """Describe positional args for check_scalar(alpha, f'alphas[{index}]', ...)."""
    return alpha, index


def witness_cd_cv_alpha_check_scalar_result(checked_alpha: object) -> object:
    """Describe alpha returned by deferred check_scalar(...)."""
    return checked_alpha
