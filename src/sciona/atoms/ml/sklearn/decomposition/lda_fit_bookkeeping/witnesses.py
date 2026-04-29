"""Ghost witnesses for LatentDirichletAllocation fit bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_lda_fit_use_online_batches(learning_method: str) -> AbstractArray:
    """Describe the Boolean online-update flag from an LDA learning method."""
    del learning_method
    return AbstractArray(shape=(), dtype="bool")


def witness_lda_fit_evaluate_iteration_due(
    iteration_index: int,
    *,
    evaluate_every: int,
) -> AbstractArray:
    """Describe the Boolean cadence check for an LDA perplexity evaluation step."""
    del iteration_index, evaluate_every
    return AbstractArray(shape=(), dtype="bool")


def witness_lda_fit_converged(
    last_bound: float | None,
    bound: float,
    *,
    perp_tol: float,
) -> AbstractArray:
    """Describe the Boolean LDA perplexity-improvement stopping predicate."""
    del last_bound, bound, perp_tol
    return AbstractArray(shape=(), dtype="bool")


def witness_lda_batch_bounds(
    n_samples: int,
    *,
    batch_size: int,
) -> AbstractArray:
    """Describe the start-stop bounds of sklearn's minibatch slices."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n_batches = (n_samples + batch_size - 1) // batch_size
    return AbstractArray(shape=(n_batches, 2), dtype="int64")


def witness_lda_partial_fit_first_call(
    *,
    has_components: bool,
) -> AbstractArray:
    """Describe the Boolean first-call flag for LDA partial_fit state."""
    del has_components
    return AbstractArray(shape=(), dtype="bool")


def witness_lda_partial_fit_require_matching_feature_count(
    n_features: int,
    *,
    trained_feature_count: int,
) -> AbstractArray:
    """Describe the validated feature-count scalar for LDA partial_fit input."""
    del trained_feature_count
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return AbstractArray(shape=(), dtype="int64", min_val=1.0)


def witness_lda_check_nonnegative_dtype_names(
    *,
    reset_n_features: bool,
    components_dtype_name: str = "float64",
) -> AbstractArray:
    """Describe the one- or two-entry dtype-name vector for LDA input validation."""
    del reset_n_features, components_dtype_name
    return AbstractArray(shape=(None,), dtype="object")
