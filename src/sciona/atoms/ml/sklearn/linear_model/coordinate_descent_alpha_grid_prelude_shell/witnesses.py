"""Ghost witnesses for sklearn coordinate-descent alpha-grid prelude atoms."""

from __future__ import annotations


def witness_cd_alpha_grid_l1_ratio_zero_guard_required(l1_ratio: object) -> object:
    """Describe the l1_ratio == 0 guard in _alpha_grid."""
    return l1_ratio


def witness_cd_alpha_grid_l1_ratio_zero_error_message(l1_ratio: object) -> object:
    """Describe the automatic-alpha-grid ValueError message for l1_ratio=0."""
    return l1_ratio


def witness_cd_alpha_grid_precomputed_Xy(Xy: object) -> object:
    """Describe the precomputed Xy branch in _alpha_grid."""
    return Xy


def witness_cd_alpha_grid_preprocess_kwargs(
    fit_intercept: object, copy_X: object, sample_weight: object
) -> object:
    """Describe the _preprocess_data kwargs assembled by _alpha_grid."""
    return fit_intercept, copy_X, sample_weight


def witness_cd_alpha_grid_yw(y: object, sample_weight: object) -> object:
    """Describe the weighted target payload used by _alpha_grid."""
    return y, sample_weight


def witness_cd_alpha_grid_dense_Xyw(X: object, yw: object) -> object:
    """Describe the dense np.dot(X.T, yw) payload in _alpha_grid."""
    return X, yw


def witness_cd_alpha_grid_sparse_mono_output_centered_Xyw(
    X: object, yw: object, X_offset: object
) -> object:
    """Describe the sparse mono-output centered Xyw payload in _alpha_grid."""
    return X, yw, X_offset
