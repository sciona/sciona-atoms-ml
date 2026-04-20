"""Ghost witnesses for selected sklearn discriminant-analysis atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import LDAState, QDAState


def witness_qda_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    priors: tuple[float, ...] | None = None,
    reg_param: float = 0.0,
    store_covariance: bool = False,
    tol: float = 1e-4,
) -> AbstractArray:
    """Describe fitting QDA class means and covariance factors."""
    del priors, store_covariance
    n_samples, n_features = _check_2d(X, "X")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if y.shape[0] != n_samples:
        raise ValueError("X and y must have matching sample counts")
    if reg_param < 0.0 or reg_param > 1.0:
        raise ValueError("reg_param must lie in [0, 1]")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_qda_decision_function(X: AbstractArray, state: QDAState) -> AbstractArray:
    """Describe QDA log posterior scores before normalization."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, state.classes.shape[0]), dtype="float64")


def witness_qda_predict_log_proba(X: AbstractArray, state: QDAState) -> AbstractArray:
    """Describe normalized log class probabilities from QDA scores."""
    return witness_qda_decision_function(X, state)


def witness_qda_predict_proba(X: AbstractArray, state: QDAState) -> AbstractArray:
    """Describe normalized class likelihood rows."""
    n_samples, _ = _check_2d(X, "X")
    return AbstractArray(shape=(n_samples, state.classes.shape[0]), dtype="float64", min_val=0.0, max_val=1.0)


def witness_qda_predict(X: AbstractArray, state: QDAState) -> AbstractArray:
    """Describe QDA class-label predictions."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def _check_2d(array: AbstractArray, name: str) -> tuple[int, int]:
    if len(array.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    return int(array.shape[0]), int(array.shape[1])


def witness_lda_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    priors: tuple[float, ...] | None = None,
    n_components: int | None = None,
    store_covariance: bool = False,
    tol: float = 1e-4,
) -> AbstractArray:
    """Describe fitting linear class means and projection weights."""
    del priors, n_components, store_covariance
    n_samples, n_features = _check_2d(X, "X")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if y.shape[0] != n_samples:
        raise ValueError("X and y must have matching sample counts")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_lda_decision_function(X: AbstractArray, state: LDAState) -> AbstractArray:
    """Describe linear discriminant confidence scores."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.classes.shape[0] == 2:
        return AbstractArray(shape=(n_samples,), dtype="float64")
    return AbstractArray(shape=(n_samples, state.classes.shape[0]), dtype="float64")


def witness_lda_predict_proba(X: AbstractArray, state: LDAState) -> AbstractArray:
    """Describe normalized linear class likelihood rows."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, state.classes.shape[0]), dtype="float64", min_val=0.0, max_val=1.0)


def witness_lda_predict_log_proba(X: AbstractArray, state: LDAState) -> AbstractArray:
    """Describe log likelihood rows."""
    return witness_lda_predict_proba(X, state)


def witness_lda_predict(X: AbstractArray, state: LDAState) -> AbstractArray:
    """Describe linear discriminant class predictions."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_lda_transform(X: AbstractArray, state: LDAState) -> AbstractArray:
    """Describe projection onto fitted class-separation axes."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, state.n_components), dtype="float64")
