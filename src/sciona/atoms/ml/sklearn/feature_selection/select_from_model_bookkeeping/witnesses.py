"""Ghost witnesses for SelectFromModel bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_select_from_model_checked_max_features(
    max_features: int,
    *,
    n_features: int,
) -> AbstractArray:
    """Describe sklearn's validated integer max_features setting."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if not 0 <= max_features <= n_features:
        raise ValueError("max_features must lie in [0, n_features]")
    return AbstractArray(shape=(), dtype="int64", min_val=0.0, max_val=float(n_features))


def witness_select_from_model_prefit_estimator_valid(
    *,
    prefit: bool,
    estimator_is_fitted: bool,
) -> AbstractArray:
    """Describe whether a prefit SelectFromModel branch accepts the estimator state."""
    del prefit, estimator_is_fitted
    return AbstractArray(shape=(), dtype="bool")


def witness_select_from_model_prefit_callable_max_features_ready(
    *,
    prefit: bool,
    max_features_is_callable: bool,
    has_fitted_max_features: bool,
) -> AbstractArray:
    """Describe whether callable max_features is usable without a prior fit-produced cache."""
    del prefit, max_features_is_callable, has_fitted_max_features
    return AbstractArray(shape=(), dtype="bool")


def witness_select_from_model_candidate_indices(
    scores: AbstractArray,
    *,
    max_features: int,
) -> AbstractArray:
    """Describe sklearn's stable descending top-k candidate index vector."""
    if len(scores.shape) != 1 or int(scores.shape[0]) < 1:
        raise ValueError("scores must be a nonempty 1D vector")
    if not 0 <= max_features <= int(scores.shape[0]):
        raise ValueError("max_features must lie in [0, len(scores)]")
    return AbstractArray(shape=(max_features,), dtype="int64")
