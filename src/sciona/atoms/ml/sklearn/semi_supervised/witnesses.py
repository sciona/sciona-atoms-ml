"""Ghost witnesses for sklearn semi-supervised atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import GraphKernel, LabelPropagationState, SelfTrainingClassifierState


def _check_2d(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return int(X.shape[0]), int(X.shape[1])


def _check_labels(y: AbstractArray, n_samples: int) -> None:
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if int(y.shape[0]) != n_samples:
        raise ValueError("X and y must have equal sample count")


def witness_label_propagation_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    kernel: GraphKernel = "rbf",
    gamma: float = 20.0,
    n_neighbors: int = 7,
    max_iter: int = 1000,
    tol: float = 1e-3,
    n_jobs: int | None = None,
) -> AbstractArray:
    """Describe fitting hard-clamped graph label propagation."""
    del gamma, n_jobs
    n_samples, _ = _check_2d(X)
    _check_labels(y, n_samples)
    if isinstance(kernel, str) and kernel not in {"rbf", "knn"}:
        raise ValueError("kernel must be 'rbf' or 'knn'")
    if not isinstance(kernel, str) and not callable(kernel):
        raise ValueError("kernel must be 'rbf', 'knn', or callable")
    if n_neighbors < 1 or max_iter < 1 or tol < 0.0:
        raise ValueError("invalid graph-label parameters")
    return AbstractArray(shape=(n_samples,), dtype="object")


def witness_label_spreading_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    kernel: GraphKernel = "rbf",
    gamma: float = 20.0,
    n_neighbors: int = 7,
    alpha: float = 0.2,
    max_iter: int = 30,
    tol: float = 1e-3,
    n_jobs: int | None = None,
) -> AbstractArray:
    """Describe fitting soft-clamped graph label spreading."""
    del gamma, n_jobs
    n_samples, _ = _check_2d(X)
    _check_labels(y, n_samples)
    if isinstance(kernel, str) and kernel not in {"rbf", "knn"}:
        raise ValueError("kernel must be 'rbf' or 'knn'")
    if not isinstance(kernel, str) and not callable(kernel):
        raise ValueError("kernel must be 'rbf', 'knn', or callable")
    if n_neighbors < 1 or max_iter < 1 or tol < 0.0 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid graph-label parameters")
    return AbstractArray(shape=(n_samples,), dtype="object")


def witness_label_propagation_predict_proba(
    X: AbstractArray,
    state: LabelPropagationState,
) -> AbstractArray:
    """Describe class probabilities from a fitted propagation state."""
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, int(state.classes.shape[0])), dtype="float64", min_val=0.0, max_val=1.0)


def witness_label_propagation_predict(
    X: AbstractArray,
    state: LabelPropagationState,
) -> AbstractArray:
    """Describe class labels from a fitted propagation state."""
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples,), dtype="object")


def witness_label_spreading_predict_proba(
    X: AbstractArray,
    state: LabelPropagationState,
) -> AbstractArray:
    """Describe class probabilities from a fitted spreading state."""
    return witness_label_propagation_predict_proba(X, state)


def witness_label_spreading_predict(
    X: AbstractArray,
    state: LabelPropagationState,
) -> AbstractArray:
    """Describe class labels from a fitted spreading state."""
    return witness_label_propagation_predict(X, state)


def witness_self_training_select_pseudo_labels(
    max_proba: AbstractArray,
    *,
    threshold: float = 0.75,
    criterion: str = "threshold",
    k_best: int = 10,
) -> AbstractArray:
    """Describe selecting confident pseudo-label candidates."""
    if len(max_proba.shape) != 1:
        raise ValueError("max_proba must be 1D")
    if not 0.0 <= threshold < 1.0:
        raise ValueError("threshold must be in [0, 1)")
    if criterion not in {"threshold", "k_best"}:
        raise ValueError("invalid self-training criterion")
    if k_best < 1:
        raise ValueError("k_best must be positive")
    return AbstractArray(shape=(int(max_proba.shape[0]),), dtype="bool")


def witness_self_training_fit(
    estimator: object,
    X: AbstractArray,
    y: AbstractArray,
    *,
    threshold: float = 0.75,
    criterion: str = "threshold",
    k_best: int = 10,
    max_iter: int | None = 10,
    verbose: bool = False,
) -> AbstractArray:
    """Describe fitting a self-training classifier and pseudo-label state."""
    del estimator, verbose
    n_samples, _ = _check_2d(X)
    _check_labels(y, n_samples)
    if not 0.0 <= threshold < 1.0:
        raise ValueError("threshold must be in [0, 1)")
    if criterion not in {"threshold", "k_best"}:
        raise ValueError("invalid self-training criterion")
    if k_best < 1:
        raise ValueError("k_best must be positive")
    if max_iter is not None and max_iter < 0:
        raise ValueError("max_iter must be non-negative or None")
    return AbstractArray(shape=(n_samples,), dtype="object")


def witness_self_training_predict(
    X: AbstractArray,
    state: SelfTrainingClassifierState,
) -> AbstractArray:
    """Describe predictions delegated to the fitted self-training estimator."""
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples,), dtype="object")


def witness_self_training_predict_proba(
    X: AbstractArray,
    state: SelfTrainingClassifierState,
) -> AbstractArray:
    """Describe probabilities delegated to the fitted self-training estimator."""
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, int(state.classes.shape[0])), dtype="float64", min_val=0.0, max_val=1.0)
