"""Ghost witnesses for sklearn multioutput chain augmentation helpers."""

from __future__ import annotations

from scipy.sparse import csr_matrix

from sciona.ghost.abstract import AbstractArray


def witness_chain_dense_cv_feature_buffer(
    X: AbstractArray,
    n_outputs: int,
) -> AbstractArray:
    """Describe dense zero-column augmentation for chain fitting with cv != None."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    if n_samples < 1 or n_features < 1 or n_outputs < 1:
        raise ValueError("X must be nonempty and n_outputs positive")
    return AbstractArray(shape=(n_samples, n_features + n_outputs), dtype="float64")


def witness_chain_sparse_cv_feature_buffer(
    X: AbstractArray,
    n_outputs: int,
) -> csr_matrix:
    """Describe sparse zero-column augmentation for chain fitting with cv != None."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    if n_samples < 1 or n_features < 1 or n_outputs < 1:
        raise ValueError("X must be nonempty and n_outputs positive")
    return csr_matrix((n_samples, n_features + n_outputs), dtype=float)


def witness_chain_sparse_training_features(
    X: AbstractArray,
    Y_ordered: AbstractArray,
) -> csr_matrix:
    """Describe sparse cv=None chain training features with ordered targets appended."""
    if len(X.shape) != 2 or len(Y_ordered.shape) != 2:
        raise ValueError("X and Y_ordered must be 2D")
    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    y_samples = int(Y_ordered.shape[0])
    n_outputs = int(Y_ordered.shape[1])
    if n_samples < 1 or n_features < 1 or y_samples != n_samples or n_outputs < 1:
        raise ValueError("X and Y_ordered must be nonempty with matching sample counts")
    return csr_matrix((n_samples, n_features + n_outputs), dtype=float)


def witness_chain_sparse_step_features(
    X: AbstractArray,
    previous_predictions: AbstractArray,
) -> csr_matrix:
    """Describe sparse prediction-step features with previous predictions appended."""
    if len(X.shape) != 2 or len(previous_predictions.shape) != 2:
        raise ValueError("X and previous_predictions must be 2D")
    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    prev_samples = int(previous_predictions.shape[0])
    n_previous = int(previous_predictions.shape[1])
    if n_samples < 1 or n_features < 1 or prev_samples != n_samples or n_previous < 1:
        raise ValueError("X and previous_predictions must be nonempty with matching sample counts")
    return csr_matrix((n_samples, n_features + n_previous), dtype=float)


def witness_chain_cv_feature_column(cv_result: AbstractArray) -> AbstractArray:
    """Describe sklearn's CV feature-column extraction for chain augmentation."""
    if len(cv_result.shape) == 1:
        n_samples = int(cv_result.shape[0])
        if n_samples < 1:
            raise ValueError("cv_result must be nonempty")
        return AbstractArray(shape=(n_samples,), dtype="float64")
    if len(cv_result.shape) == 2:
        n_samples = int(cv_result.shape[0])
        n_columns = int(cv_result.shape[1])
        if n_samples < 1 or n_columns < 2:
            raise ValueError("2D cv_result must have at least two columns")
        return AbstractArray(shape=(n_samples,), dtype="float64")
    raise ValueError("cv_result must be 1D or 2D")
