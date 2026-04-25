"""Ghost witnesses for SelectorMixin sparse inverse-transform helpers."""

from __future__ import annotations

from scipy.sparse import csc_matrix

from sciona.ghost.abstract import AbstractArray


def witness_selector_inverse_transform_csc(
    X_selected: AbstractArray,
    support_mask: AbstractArray,
) -> csc_matrix:
    """Describe sparse inverse feature selection with zero-filled dropped columns."""
    if len(support_mask.shape) != 1:
        raise ValueError("support_mask must be 1D")
    n_features = int(support_mask.shape[0])
    if n_features < 1:
        raise ValueError("support_mask must be nonempty")
    if len(X_selected.shape) != 2:
        raise ValueError("X_selected must be 2D")
    n_samples = int(X_selected.shape[0])
    if n_samples < 1:
        raise ValueError("X_selected must be nonempty")
    return csc_matrix((n_samples, n_features), dtype=float)
