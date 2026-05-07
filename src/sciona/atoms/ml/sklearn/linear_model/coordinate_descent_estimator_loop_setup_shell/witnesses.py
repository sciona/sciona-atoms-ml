"""Ghost witnesses for sklearn coordinate-descent estimator loop setup atoms."""

from __future__ import annotations


def witness_cd_estimator_initial_coef_required(warm_start: object, has_coef_attr: object) -> object:
    """Describe whether ElasticNet.fit allocates fresh coefficients."""
    return warm_start, has_coef_attr


def witness_cd_estimator_initial_coef_zeros(
    n_targets: object, n_features: object, dtype: object
) -> object:
    """Describe fresh coefficient allocation."""
    return n_targets, n_features, dtype


def witness_cd_estimator_warm_start_coef_matrix(coef: object) -> object:
    """Describe warm-start coefficient matrix normalization."""
    return coef


def witness_cd_estimator_dual_gaps_zeros(n_targets: object, dtype: object) -> object:
    """Describe dual-gaps buffer allocation."""
    return n_targets, dtype


def witness_cd_estimator_n_iter_list_initial(n_targets: object) -> object:
    """Describe initial n_iter_ list assignment."""
    return n_targets


def witness_cd_estimator_loop_this_xy(Xy: object, target_index: object) -> object:
    """Describe per-target Xy selection."""
    return Xy, target_index


def witness_cd_estimator_single_alpha_grid(alpha: object) -> object:
    """Describe one-alpha list passed to path(...)."""
    return alpha


def witness_cd_estimator_path_args(X: object, y: object, target_index: object) -> object:
    """Describe positional args for one self.path(...) callback."""
    return X, y, target_index


def witness_cd_estimator_path_kwargs(
    l1_ratio: object,
    alpha: object,
    precompute: object,
    this_Xy: object,
    coef_init: object,
    positive: object,
    tol: object,
    X_offset: object,
    X_scale: object,
    max_iter: object,
    random_state: object,
    selection: object,
    sample_weight: object,
) -> object:
    """Describe kwargs for one self.path(...) callback."""
    return (
        l1_ratio,
        alpha,
        precompute,
        this_Xy,
        coef_init,
        positive,
        tol,
        X_offset,
        X_scale,
        max_iter,
        random_state,
        selection,
        sample_weight,
    )
