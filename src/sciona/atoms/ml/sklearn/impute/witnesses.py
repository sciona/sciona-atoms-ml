"""Ghost witnesses for sklearn imputation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import KNNImputerState, MissingIndicatorState, SimpleImputerState


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


def witness_nan_euclidean_distances(X: AbstractArray, Y: AbstractArray) -> AbstractArray:
    """Describe pairwise nan-aware Euclidean distances."""
    if len(X.shape) != 2 or len(Y.shape) != 2:
        raise ValueError("X and Y must be 2D")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have matching feature counts")
    return AbstractArray(shape=(int(X.shape[0]), int(Y.shape[0])), dtype="float64")


def witness_knn_imputer_calc_impute(
    dist_pot_donors: AbstractArray,
    fit_X_col: AbstractArray,
    mask_fit_X_col: AbstractArray,
    *,
    n_neighbors: int,
    weights: str = "uniform",
) -> AbstractArray:
    """Describe donor averaging for one imputed feature column."""
    if len(dist_pot_donors.shape) != 2:
        raise ValueError("distances must be 2D")
    if len(fit_X_col.shape) != 1 or len(mask_fit_X_col.shape) != 1:
        raise ValueError("donor values and masks must be 1D")
    if fit_X_col.shape[0] != mask_fit_X_col.shape[0]:
        raise ValueError("donor values and masks must have matching length")
    if dist_pot_donors.shape[1] != fit_X_col.shape[0]:
        raise ValueError("distance columns must match donor values")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if weights not in {"uniform", "distance"}:
        raise ValueError("weights is unsupported")
    return AbstractArray(shape=(int(dist_pot_donors.shape[0]),), dtype="float64")


def witness_knn_imputer_fit(
    X: AbstractArray,
    *,
    n_neighbors: int = 5,
    weights: str = "uniform",
    keep_empty_features: bool = False,
) -> AbstractArray:
    """Describe training state learned by dense KNN imputation."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if weights not in {"uniform", "distance"}:
        raise ValueError("weights is unsupported")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_knn_imputer_transform(X: AbstractArray, state: KNNImputerState) -> AbstractArray:
    """Describe dense KNN imputation with a fitted training matrix."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.fit_X.shape[1]:
        raise ValueError("X feature count must match fitted state")
    output_features = int(state.fit_X.shape[1]) if state.keep_empty_features else int(state.valid_mask.sum())
    return AbstractArray(shape=(int(X.shape[0]), output_features), dtype="float64")
