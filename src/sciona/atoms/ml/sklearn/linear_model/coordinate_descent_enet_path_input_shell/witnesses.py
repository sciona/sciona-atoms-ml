"""Ghost witnesses for sklearn coordinate-descent enet_path input-shell atoms."""

from __future__ import annotations


def witness_cd_enet_path_unexpected_params_guard_required(params: object) -> object:
    """Describe the `if len(params) > 0:` guard in enet_path."""
    return params


def witness_cd_enet_path_check_input_branch(check_input: object) -> object:
    """Describe the `if check_input:` branch predicate in enet_path."""
    return check_input


def witness_cd_enet_path_Xy_validation_required(Xy: object) -> object:
    """Describe the `if Xy is not None:` validation gate in enet_path."""
    return Xy


def witness_cd_enet_path_sparse_scaling_required(
    multi_output: object, x_is_sparse: object
) -> object:
    """Describe the sparse-scaling branch predicate in enet_path."""
    return multi_output, x_is_sparse


def witness_cd_enet_path_sparse_scaling(
    X_offset_param: object, X_scale_param: object, n_features: object, dtype_name: object
) -> object:
    """Describe the X_sparse_scaling shell in enet_path."""
    return X_offset_param, X_scale_param, n_features, dtype_name


def witness_cd_enet_path_prefit_kwargs(check_input: object) -> object:
    """Describe the fixed _pre_fit kwargs shell in enet_path."""
    return check_input


def witness_cd_enet_path_alpha_grid_required(alphas: object) -> object:
    """Describe the `if alphas is None:` branch predicate in enet_path."""
    return alphas
