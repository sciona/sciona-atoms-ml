"""Ghost witnesses for sklearn kernel approximation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import RBFSamplerState


def witness_rbf_sampler_fit(
    X: AbstractArray,
    *,
    gamma: float | str = 1.0,
    n_components: int = 100,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe fitting random Fourier weights for RBF features."""
    del random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if not (gamma == "scale" or (isinstance(gamma, (int, float)) and not isinstance(gamma, bool) and gamma > 0.0)):
        raise ValueError("gamma must be positive or 'scale'")
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    return AbstractArray(shape=(int(X.shape[1]), n_components), dtype="float64")


def witness_rbf_sampler_transform(X: AbstractArray, state: RBFSamplerState) -> AbstractArray:
    """Describe projecting samples into fitted RBF random features."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), state.n_components), dtype="float64")
