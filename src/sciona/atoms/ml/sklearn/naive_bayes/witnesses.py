"""Ghost witnesses for sklearn naive Bayes atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import GaussianNBState


def witness_gaussian_nb_update_mean_variance(
    n_past: float,
    mu: AbstractArray,
    var: AbstractArray,
    X: AbstractArray,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe one online Gaussian mean and variance update."""
    if n_past < 0:
        raise ValueError("n_past must be nonnegative")
    if len(mu.shape) != 1 or len(var.shape) != 1:
        raise ValueError("mu and var must be 1D")
    if mu.shape != var.shape:
        raise ValueError("mu and var must have matching shape")
    if len(X.shape) != 2 or X.shape[1] != mu.shape[0]:
        raise ValueError("X must be 2D with the same feature count as mu")
    if sample_weight is not None and sample_weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must match the row count of X")
    return AbstractArray(shape=(2, int(mu.shape[0])), dtype="float64")


def witness_gaussian_nb_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    priors: AbstractArray | None = None,
    var_smoothing: float = 1e-9,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe dense Gaussian naive Bayes state learned from labels."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if y.shape != (X.shape[0],):
        raise ValueError("y must match the row count of X")
    if priors is not None and len(priors.shape) != 1:
        raise ValueError("priors must be 1D")
    if sample_weight is not None and sample_weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must match the row count of X")
    if var_smoothing < 0:
        raise ValueError("var_smoothing must be nonnegative")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_gaussian_nb_joint_log_likelihood(X: AbstractArray, state: GaussianNBState) -> AbstractArray:
    """Describe Gaussian class joint log likelihoods for each row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), int(state.classes.shape[0])), dtype="float64")


def witness_gaussian_nb_predict_log_proba(X: AbstractArray, state: GaussianNBState) -> AbstractArray:
    """Describe normalized Gaussian log probabilities for each class."""
    return witness_gaussian_nb_joint_log_likelihood(X, state)


def witness_gaussian_nb_predict_proba(X: AbstractArray, state: GaussianNBState) -> AbstractArray:
    """Describe normalized Gaussian probabilities for each class."""
    return witness_gaussian_nb_joint_log_likelihood(X, state)


def witness_gaussian_nb_predict(X: AbstractArray, state: GaussianNBState) -> AbstractArray:
    """Describe one integer class prediction per input row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="int64")
