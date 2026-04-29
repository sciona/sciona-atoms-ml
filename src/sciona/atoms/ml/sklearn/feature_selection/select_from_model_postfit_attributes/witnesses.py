"""Ghost witnesses for SelectFromModel post-fit attribute helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_select_from_model_partial_fit_first_call(
    *,
    estimator_is_initialized: bool,
) -> AbstractArray:
    """Describe sklearn's SelectFromModel.partial_fit first-call flag."""
    del estimator_is_initialized
    return AbstractArray(shape=(), dtype="bool")


def witness_select_from_model_postfit_n_features_in(
    n_features_in: int,
) -> AbstractArray:
    """Describe the fitted n_features_in_ value copied from the wrapped estimator."""
    if n_features_in < 1:
        raise ValueError("n_features_in must be positive")
    return AbstractArray(shape=(), dtype="int64", min_val=1.0)


def witness_select_from_model_postfit_feature_names_in(
    feature_names_in: tuple[str, ...],
) -> tuple[str, ...]:
    """Describe the fitted feature_names_in_ tuple copied from the wrapped estimator."""
    if len(feature_names_in) < 1:
        raise ValueError("feature_names_in must be nonempty")
    return feature_names_in
