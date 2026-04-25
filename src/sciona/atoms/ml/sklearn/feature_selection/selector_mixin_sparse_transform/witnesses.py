"""Ghost witnesses for SelectorMixin sparse transform helpers."""

from __future__ import annotations

from scipy.sparse import csr_matrix

from sciona.ghost.abstract import AbstractArray


def witness_selector_transform_sparse(
    X: AbstractArray,
    support_mask: AbstractArray,
) -> csr_matrix | AbstractArray:
    """Describe sparse feature selection with the no-selected-features fallback."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(support_mask.shape) != 1:
        raise ValueError("support_mask must be 1D")
    if int(X.shape[1]) != int(support_mask.shape[0]):
        raise ValueError("X and support_mask must agree on feature count")
    n_samples = int(X.shape[0])
    n_features = int(support_mask.shape[0])
    if n_samples < 1 or n_features < 1:
        raise ValueError("X and support_mask must be nonempty")
    return csr_matrix((n_samples, n_features), dtype=float)
