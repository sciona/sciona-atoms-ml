"""Ghost witnesses for sklearn imputation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import MissingIndicatorState, SimpleImputerState


def witness_simple_imputer_fit(
    X: AbstractArray,
    *,
    strategy: str = "mean",
    fill_value: float = 0.0,
    keep_empty_features: bool = False,
) -> AbstractArray:
    """Describe per-feature statistics learned by a simple imputer."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if strategy not in {"mean", "median", "most_frequent", "constant"}:
        raise ValueError("strategy is unsupported")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")


def witness_simple_imputer_transform(X: AbstractArray, state: SimpleImputerState) -> AbstractArray:
    """Describe dense column imputation with fitted statistics."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), int(state.valid_features.shape[0])), dtype="float64")


def witness_missing_indicator_fit(X: AbstractArray, *, features: str = "missing-only") -> AbstractArray:
    """Describe feature indices selected for missing-value indicators."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if features not in {"missing-only", "all"}:
        raise ValueError("features is unsupported")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="int64")


def witness_missing_indicator_transform(
    X: AbstractArray,
    state: MissingIndicatorState,
    *,
    error_on_new: bool = True,
) -> AbstractArray:
    """Describe a boolean missing-value mask selected by fitted features."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), int(state.features.shape[0])), dtype="bool")
