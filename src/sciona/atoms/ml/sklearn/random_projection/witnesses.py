"""Ghost witnesses for sklearn random projection atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import RandomProjectionState


def _valid_n_components(n_components: int | str) -> bool:
    return n_components == "auto" or (isinstance(n_components, int) and n_components >= 1)


def _valid_density(density: float | str) -> bool:
    return density == "auto" or (isinstance(density, (float, int)) and 0.0 < float(density) <= 1.0)


def witness_gaussian_random_projection_fit(
    X: AbstractArray,
    n_components: int | str = "auto",
    eps: float = 0.1,
    compute_inverse_components: bool = False,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe the learned Gaussian projection matrix."""
    del compute_inverse_components, random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[0] < 1:
        raise ValueError("X must contain at least one sample")
    if not _valid_n_components(n_components):
        raise ValueError("n_components must be a positive integer or 'auto'")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    out_features = X.shape[1] if n_components == "auto" else int(n_components)
    return AbstractArray(shape=(out_features, X.shape[1]), dtype=X.dtype)


def witness_gaussian_random_projection_transform(
    X: AbstractArray,
    state: RandomProjectionState,
) -> AbstractArray:
    """Describe dense Gaussian random projection output."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.projection_kind != "gaussian":
        raise ValueError("state must be Gaussian projection state")
    return AbstractArray(shape=(X.shape[0], state.n_components), dtype=X.dtype)


def witness_sparse_random_projection_fit(
    X: AbstractArray,
    n_components: int | str = "auto",
    density: float | str = "auto",
    eps: float = 0.1,
    dense_output: bool = False,
    compute_inverse_components: bool = False,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe the learned sparse projection matrix."""
    del dense_output, compute_inverse_components, random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[0] < 1:
        raise ValueError("X must contain at least one sample")
    if not _valid_n_components(n_components):
        raise ValueError("n_components must be a positive integer or 'auto'")
    if not _valid_density(density):
        raise ValueError("density must be 'auto' or in (0, 1]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    out_features = X.shape[1] if n_components == "auto" else int(n_components)
    return AbstractArray(shape=(out_features, X.shape[1]), dtype=X.dtype)


def witness_sparse_random_projection_transform(
    X: AbstractArray,
    state: RandomProjectionState,
) -> AbstractArray:
    """Describe sparse random projection output."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.projection_kind != "sparse":
        raise ValueError("state must be sparse projection state")
    return AbstractArray(shape=(X.shape[0], state.n_components), dtype=X.dtype)


def witness_random_projection_inverse_transform(
    X: AbstractArray,
    state: RandomProjectionState,
) -> AbstractArray:
    """Describe reconstruction from random projection coordinates."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_components:
        raise ValueError("X feature count must match fitted projection components")
    return AbstractArray(shape=(X.shape[0], state.n_features_in), dtype=X.dtype)
