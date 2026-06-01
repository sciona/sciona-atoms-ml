from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_center_data(X: AbstractArray) -> AbstractArray:
    """Ghost witness for center_data."""
    _ = (X)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_pca_svd_decompose(X_centered: AbstractArray, n_components: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for pca_svd_decompose."""
    _ = (X_centered, n_components)
    return AbstractArray(shape=X_centered.shape, dtype=X_centered.dtype)

def witness_calculate_pca_variance(singular_values: AbstractArray, n_samples: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for calculate_pca_variance."""
    _ = (singular_values, n_samples)
    return AbstractArray(shape=singular_values.shape, dtype=singular_values.dtype)

