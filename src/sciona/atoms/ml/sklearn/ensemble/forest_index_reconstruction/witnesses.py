"""Ghost witnesses for sklearn forest index-reconstruction helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_forest_estimator_sample_indices(
    bootstrap: bool,
    n_samples: int,
    n_samples_bootstrap: int | None,
    random_state: int,
) -> AbstractArray:
    """Describe the in-bag sample indices for one forest tree."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if random_state < 0:
        raise ValueError("random_state must be nonnegative")
    if bootstrap:
        if n_samples_bootstrap is None or n_samples_bootstrap < 1:
            raise ValueError("bootstrap draws need a positive bootstrap sample count")
        return AbstractArray(shape=(n_samples_bootstrap,), dtype="int64")
    if n_samples_bootstrap is not None:
        raise ValueError("n_samples_bootstrap must be None when bootstrap is disabled")
    return AbstractArray(shape=(n_samples,), dtype="int64")


def witness_forest_estimators_sample_indices(
    random_states: tuple[int, ...],
    bootstrap: bool,
    n_samples: int,
    n_samples_bootstrap: int | None,
) -> tuple[AbstractArray, ...]:
    """Describe the in-bag sample-index vectors for all fitted forest trees."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if any(state < 0 for state in random_states):
        raise ValueError("random_states must be nonnegative")
    return tuple(
        witness_forest_estimator_sample_indices(
            bootstrap,
            n_samples,
            n_samples_bootstrap,
            state,
        )
        for state in random_states
    )
