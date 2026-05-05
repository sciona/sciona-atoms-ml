"""Ghost witnesses for sklearn coordinate-descent enet_path bookkeeping atoms."""

from __future__ import annotations


def witness_cd_enet_path_multi_output(y_ndim: object) -> object:
    """Describe the `multi_output = y.ndim != 1` shell in enet_path."""
    return y_ndim


def witness_cd_enet_path_target_count(y_shape: object, multi_output: object) -> object:
    """Describe the `n_targets = y.shape[1]` shell in enet_path."""
    return y_shape, multi_output


def witness_cd_enet_path_positive_multi_output_guard_required(
    multi_output: object, positive: object
) -> object:
    """Describe the `if multi_output and positive:` guard in enet_path."""
    return multi_output, positive


def witness_cd_enet_path_sorted_alphas(alphas: object) -> object:
    """Describe the descending alpha sort shell in enet_path."""
    return alphas


def witness_cd_enet_path_random_selection(selection: object) -> object:
    """Describe the `random = selection == \"random\"` shell in enet_path."""
    return selection


def witness_cd_enet_path_regularization_pair(
    alpha: object, l1_ratio: object, n_samples: object
) -> object:
    """Describe the l1_reg/l2_reg scaling shell in enet_path."""
    return alpha, l1_ratio, n_samples


def witness_cd_enet_path_outputs(
    alphas: object, coefs: object, dual_gaps: object, n_iters: object, return_n_iter: object
) -> object:
    """Describe the final return packaging shell in enet_path."""
    return alphas, coefs, dual_gaps, n_iters, return_n_iter
