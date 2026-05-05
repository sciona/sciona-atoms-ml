"""Ghost witnesses for sklearn coordinate-descent alpha-grid math atoms."""

from __future__ import annotations


def witness_cd_alpha_grid_xyw_matrix(Xyw: object) -> object:
    """Describe the `if Xyw.ndim == 1: Xyw = Xyw[:, np.newaxis]` shell in _alpha_grid."""
    return Xyw


def witness_cd_alpha_grid_sample_count(
    n_samples: object, sample_weight_sum: object
) -> object:
    """Describe the sample-count shell in _alpha_grid."""
    return n_samples, sample_weight_sum


def witness_cd_alpha_grid_alpha_max(
    Xyw: object, sample_count: object, l1_ratio: object
) -> object:
    """Describe the alpha_max computation in _alpha_grid."""
    return Xyw, sample_count, l1_ratio


def witness_cd_alpha_grid_use_resolution_fallback(alpha_max: object) -> object:
    """Describe the floating-point resolution fallback predicate in _alpha_grid."""
    return alpha_max


def witness_cd_alpha_grid_values(
    alpha_max: object, eps: object, n_alphas: object
) -> object:
    """Describe the final geomspace/fallback alpha grid in _alpha_grid."""
    return alpha_max, eps, n_alphas
