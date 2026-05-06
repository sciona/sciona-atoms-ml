"""Ghost witnesses for sklearn coordinate-descent path-residual mono-output normalization atoms."""

from __future__ import annotations


def witness_cd_path_residuals_use_mono_output_normalization(y_ndim: object) -> object:
    """Describe the y.ndim == 1 normalization branch in _path_residuals."""
    return y_ndim


def witness_cd_path_residuals_mono_output_coefs(coefs: object) -> object:
    """Describe the coefs[np.newaxis, :, :] normalization in _path_residuals."""
    return coefs


def witness_cd_path_residuals_mono_output_y_offset(y_offset: object) -> object:
    """Describe the np.atleast_1d(y_offset) normalization in _path_residuals."""
    return y_offset


def witness_cd_path_residuals_mono_output_y_test(y_test: object) -> object:
    """Describe the y_test[:, np.newaxis] normalization in _path_residuals."""
    return y_test
