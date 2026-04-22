"""Ghost witnesses for sklearn Neighborhood Components Analysis helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 2:
        raise ValueError(f"{name} must contain at least two samples")
    return size


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 2 or cols < 1:
        raise ValueError(f"{name} must have at least two rows and one column")
    return rows, cols


def witness_nca_same_class_mask(y: AbstractArray) -> AbstractArray:
    """Describe the fixed same-class mask used by NCA optimization."""
    n_samples = _check_vector(y, "y")
    return AbstractArray(shape=(n_samples, n_samples), dtype="bool")


def witness_nca_linear_transform(X: AbstractArray, components: AbstractArray) -> AbstractArray:
    """Describe applying an NCA component matrix to samples."""
    n_samples, n_features = _check_matrix(X, "X")
    n_components, component_features = _check_matrix(components, "components")
    if component_features != n_features:
        raise ValueError("component feature count must match X")
    return AbstractArray(shape=(n_samples, n_components), dtype="float64")


def witness_nca_neighbor_probabilities(X_embedded: AbstractArray) -> AbstractArray:
    """Describe NCA row-wise neighbor probabilities in embedded space."""
    n_samples, _ = _check_matrix(X_embedded, "X_embedded")
    return AbstractArray(shape=(n_samples, n_samples), dtype="float64")


def witness_nca_loss_gradient(
    transformation: AbstractArray,
    X: AbstractArray,
    same_class_mask: AbstractArray,
    *,
    sign: float = 1.0,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe the NCA objective value and flattened gradient."""
    del sign
    n_transform = _check_vector(transformation, "transformation")
    n_samples, n_features = _check_matrix(X, "X")
    mask_rows, mask_cols = _check_matrix(same_class_mask, "same_class_mask")
    if mask_rows != n_samples or mask_cols != n_samples:
        raise ValueError("same_class_mask must be square over X samples")
    if n_transform % n_features != 0:
        raise ValueError("transformation length must be a multiple of X features")
    return (
        AbstractArray(shape=(), dtype="float64"),
        AbstractArray(shape=(n_transform,), dtype="float64"),
    )
