"""Ghost witnesses for sparse-encode preprocessing atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_sparse_encode_regularization(
    algorithm: str,
    *,
    n_features: int,
    n_components: int,
    n_nonzero_coefs: int | None = None,
    alpha: float | None = None,
) -> float:
    """Describe sparse-encode regularization selection."""
    del n_nonzero_coefs, alpha
    if algorithm not in {"lasso_lars", "lasso_cd", "lars", "omp", "threshold"}:
        raise ValueError("unsupported algorithm")
    if n_features < 1 or n_components < 1:
        raise ValueError("feature and component counts must be positive")
    return 0.0


def witness_sparse_encode_gram(dictionary: AbstractArray) -> AbstractArray:
    """Describe the dictionary Gram matrix."""
    n_components, _ = _check_matrix(dictionary, "dictionary")
    return AbstractArray(shape=(n_components, n_components), dtype="float64")


def witness_sparse_encode_covariance(X: AbstractArray, dictionary: AbstractArray) -> AbstractArray:
    """Describe the sparse-encode covariance matrix."""
    n_samples, n_features = _check_matrix(X, "X")
    n_components, dict_features = _check_matrix(dictionary, "dictionary")
    if dict_features != n_features:
        raise ValueError("feature counts must match")
    return AbstractArray(shape=(n_components, n_samples), dtype="float64")


def witness_sparse_encode_threshold(
    X: AbstractArray,
    dictionary: AbstractArray,
    *,
    cov: AbstractArray | None = None,
    alpha: float | None = None,
    positive: bool = False,
) -> AbstractArray:
    """Describe threshold sparse-encode output."""
    del alpha, positive
    n_samples, n_features = _check_matrix(X, "X")
    n_components, dict_features = _check_matrix(dictionary, "dictionary")
    if dict_features != n_features:
        raise ValueError("feature counts must match")
    if cov is not None and cov.shape != (n_components, n_samples):
        raise ValueError("cov must be components by samples")
    return AbstractArray(shape=(n_samples, n_components), dtype="float64")
