"""Ghost witnesses for sklearn forest prediction preflight helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_forest_predict_ensure_all_finite_mode(
    supports_missing_values: bool,
) -> str | bool:
    """Describe the ensure-all-finite mode used before forest prediction."""
    return "allow-nan" if supports_missing_values else True


def witness_forest_predict_require_sparse_int32_indices(
    indices: AbstractArray,
    indptr: AbstractArray,
) -> bool:
    """Describe the sparse index dtype guard used before forest prediction."""
    if len(indices.shape) != 1 or len(indptr.shape) != 1:
        raise ValueError("indices and indptr must be 1D")
    return True
