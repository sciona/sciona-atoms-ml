"""Ghost witnesses for sklearn multioutput chain fit-step data helpers."""

from __future__ import annotations

from scipy.sparse import csr_matrix

from sciona.ghost.abstract import AbstractArray


def witness_chain_fit_step_feature_limit(
    n_base_features: int,
    chain_idx: int,
) -> int:
    """Describe the feature-prefix width used by sklearn for one chain fit step."""
    if n_base_features < 1 or chain_idx < 0:
        raise ValueError("n_base_features must be positive and chain_idx nonnegative")
    return int


def witness_chain_fit_target_column(
    Y: AbstractArray,
    order: AbstractArray,
    chain_idx: int,
) -> AbstractArray:
    """Describe one ordered target column selected for a chain fit step."""
    if len(Y.shape) != 2 or len(order.shape) != 1:
        raise ValueError("Y must be 2D and order must be 1D")
    n_samples = int(Y.shape[0])
    n_outputs = int(Y.shape[1])
    if n_samples < 1 or n_outputs < 1 or int(order.shape[0]) != n_outputs:
        raise ValueError("Y and order must be nonempty with matching output count")
    if chain_idx < 0 or chain_idx >= n_outputs:
        raise ValueError("chain_idx must select an existing output")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_chain_fit_dense_step_features(
    X_aug: AbstractArray,
    feature_limit: int,
) -> AbstractArray:
    """Describe the dense augmented feature prefix used for one chain fit step."""
    if len(X_aug.shape) != 2:
        raise ValueError("X_aug must be 2D")
    n_samples = int(X_aug.shape[0])
    n_features = int(X_aug.shape[1])
    if n_samples < 1 or n_features < 1 or feature_limit < 1 or feature_limit > n_features:
        raise ValueError("X_aug must be nonempty and feature_limit must select a valid prefix")
    return AbstractArray(shape=(n_samples, feature_limit), dtype="float64")


def witness_chain_fit_sparse_step_features(
    X_aug: AbstractArray,
    feature_limit: int,
) -> csr_matrix:
    """Describe the sparse augmented feature prefix used for one chain fit step."""
    if len(X_aug.shape) != 2:
        raise ValueError("X_aug must be 2D")
    n_samples = int(X_aug.shape[0])
    n_features = int(X_aug.shape[1])
    if n_samples < 1 or n_features < 1 or feature_limit < 1 or feature_limit > n_features:
        raise ValueError("X_aug must be nonempty and feature_limit must select a valid prefix")
    return csr_matrix((n_samples, feature_limit), dtype=float)
