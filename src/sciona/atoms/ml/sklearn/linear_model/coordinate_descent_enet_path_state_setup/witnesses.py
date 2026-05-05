"""Ghost witnesses for sklearn coordinate-descent enet_path state-setup atoms."""

from __future__ import annotations


def witness_cd_enet_path_alpha_count(alphas: object) -> object:
    """Describe the `n_alphas = len(alphas)` shell in enet_path."""
    return alphas


def witness_cd_enet_path_dual_gap_buffer(alpha_count: object) -> object:
    """Describe the `dual_gaps = np.empty(n_alphas)` shell in enet_path."""
    return alpha_count


def witness_cd_enet_path_iteration_buffer(alpha_count: object) -> object:
    """Describe the `n_iters = []` shell in enet_path."""
    return alpha_count


def witness_cd_enet_path_coef_path_shape(
    n_features: object, alpha_count: object, multi_output: object, n_targets: object
) -> object:
    """Describe the coefficient-path allocation shape shell in enet_path."""
    return n_features, alpha_count, multi_output, n_targets


def witness_cd_enet_path_coef_path_buffer(coef_shape: object, dtype_name: object) -> object:
    """Describe the `coefs = np.empty(..., dtype=X.dtype)` shell in enet_path."""
    return coef_shape, dtype_name


def witness_cd_enet_path_initial_coef_required(coef_init: object) -> object:
    """Describe the `if coef_init is None:` branch predicate in enet_path."""
    return coef_init


def witness_cd_enet_path_initial_coef(
    coef_shape: object, dtype_name: object, coef_init: object
) -> object:
    """Describe the initial coefficient buffer shell in enet_path."""
    return coef_shape, dtype_name, coef_init
