"""Ghost witnesses for sklearn coordinate-descent path-residual prelude atoms."""

from __future__ import annotations


def witness_cd_path_residuals_use_sample_weight_branch(sample_weight: object) -> object:
    """Describe the sample-weight presence branch in _path_residuals."""
    return sample_weight


def witness_cd_path_residuals_train_sample_weight(sample_weight: object, train: object) -> object:
    """Describe the training sample-weight slicing shell in _path_residuals."""
    return sample_weight, train


def witness_cd_path_residuals_test_sample_weight(sample_weight: object, test: object) -> object:
    """Describe the test sample-weight slicing shell in _path_residuals."""
    return sample_weight, test


def witness_cd_path_residuals_train_sample_count(x_train_shape: object) -> object:
    """Describe the training sample-count shell in _path_residuals."""
    return x_train_shape


def witness_cd_path_residuals_rescaled_train_sample_weight(
    train_sample_weight: object, train_sample_count: object
) -> object:
    """Describe the train sample-weight rescaling shell in _path_residuals."""
    return train_sample_weight, train_sample_count


def witness_cd_path_residuals_use_gram_precompute(y_ndim: object) -> object:
    """Describe the mono-output precompute branch in _path_residuals."""
    return y_ndim


def witness_cd_path_residuals_resolved_precompute(
    path_precompute: object, use_gram_precompute: object
) -> object:
    """Describe the resolved precompute value in _path_residuals."""
    return path_precompute, use_gram_precompute
