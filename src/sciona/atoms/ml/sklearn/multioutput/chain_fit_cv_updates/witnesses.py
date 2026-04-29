"""Ghost witnesses for sklearn multioutput chain fit-time CV updates."""

from __future__ import annotations

from scipy.sparse import lil_matrix

from sciona.ghost.abstract import AbstractArray


def witness_chain_fit_cv_update_required(
    chain_idx: int,
    n_outputs: int,
) -> bool:
    """Describe whether sklearn updates a chain CV feature column at this step."""
    if chain_idx < 0 or n_outputs < 2 or chain_idx >= n_outputs:
        raise ValueError("chain_idx must select a valid output within a multi-output chain")
    return bool


def witness_chain_fit_feature_column_index(
    n_base_features: int,
    chain_idx: int,
) -> int:
    """Describe the fit-time augmented feature-column index for one chain step."""
    if n_base_features < 1 or chain_idx < 0:
        raise ValueError("n_base_features must be positive and chain_idx must be nonnegative")
    return int


def witness_chain_fit_dense_cv_feature_update(
    X_aug: AbstractArray,
    cv_column: AbstractArray,
    col_idx: int,
) -> AbstractArray:
    """Describe dense CV feature-column assignment during chain fitting."""
    if len(X_aug.shape) != 2 or len(cv_column.shape) != 1:
        raise ValueError("X_aug must be 2D and cv_column must be 1D")
    n_samples = int(X_aug.shape[0])
    n_features = int(X_aug.shape[1])
    if n_samples < 1 or n_features < 1 or int(cv_column.shape[0]) != n_samples:
        raise ValueError("X_aug and cv_column must be nonempty with matching sample counts")
    if col_idx < 0 or col_idx >= n_features:
        raise ValueError("col_idx must select an existing augmented feature column")
    return AbstractArray(shape=(n_samples, n_features), dtype="float64")


def witness_chain_fit_sparse_cv_feature_update(
    X_aug: AbstractArray,
    cv_column: AbstractArray,
    col_idx: int,
) -> lil_matrix:
    """Describe sparse CV feature-column assignment during chain fitting."""
    if len(X_aug.shape) != 2 or len(cv_column.shape) != 1:
        raise ValueError("X_aug must be 2D and cv_column must be 1D")
    n_samples = int(X_aug.shape[0])
    n_features = int(X_aug.shape[1])
    if n_samples < 1 or n_features < 1 or int(cv_column.shape[0]) != n_samples:
        raise ValueError("X_aug and cv_column must be nonempty with matching sample counts")
    if col_idx < 0 or col_idx >= n_features:
        raise ValueError("col_idx must select an existing augmented feature column")
    return lil_matrix((n_samples, n_features), dtype=float)
