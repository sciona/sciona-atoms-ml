"""Ghost witnesses for sklearn multi-output solver setup atoms."""

from __future__ import annotations


def witness_cd_multitask_preprocess_data_args(X: object, y: object) -> object:
    """Describe _preprocess_data positional arguments."""
    return X, y


def witness_cd_multitask_preprocess_data_kwargs(fit_intercept: object) -> object:
    """Describe _preprocess_data keyword arguments."""
    return fit_intercept


def witness_cd_multitask_fresh_coef_required(
    warm_start: object, has_coef_attr: object
) -> object:
    """Describe whether fresh coefficients are allocated."""
    return warm_start, has_coef_attr


def witness_cd_multitask_initial_coef_zeros(
    n_targets: object, n_features: object, dtype: object
) -> object:
    """Describe fresh coefficient allocation."""
    return n_targets, n_features, dtype


def witness_cd_multitask_regularization(
    alpha: object, l1_ratio: object, n_samples: object
) -> object:
    """Describe l1 and l2 regularization scaling."""
    return alpha, l1_ratio, n_samples


def witness_cd_multitask_coef_fortran_array(coef: object) -> object:
    """Describe coefficient Fortran-order normalization."""
    return coef


def witness_cd_multitask_random_state_args(random_state: object) -> object:
    """Describe check_random_state positional arguments."""
    return random_state


def witness_cd_multitask_solver_args(
    coef: object,
    l1_reg: object,
    l2_reg: object,
    X: object,
    y: object,
    max_iter: object,
    tol: object,
    checked_random_state: object,
    random: object,
) -> object:
    """Describe compiled multitask solver positional arguments."""
    return (
        coef,
        l1_reg,
        l2_reg,
        X,
        y,
        max_iter,
        tol,
        checked_random_state,
        random,
    )
