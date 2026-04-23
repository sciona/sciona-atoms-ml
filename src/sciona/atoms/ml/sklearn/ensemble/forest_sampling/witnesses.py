"""Ghost witnesses for sklearn forest bootstrap sampling atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_forest_resolve_bootstrap_sample_count(
    n_samples: int,
    max_samples: int | float | None,
) -> int:
    """Describe the scalar bootstrap draw count implied by a validated max_samples setting."""
    del max_samples
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return int(n_samples)


def witness_forest_generate_sample_indices(
    n_samples: int,
    n_samples_bootstrap: int,
    *,
    random_state: int | object | None = None,
) -> AbstractArray:
    """Describe one vector of bootstrap sample indices."""
    del random_state
    if n_samples < 1 or n_samples_bootstrap < 1:
        raise ValueError("sample counts must be positive")
    return AbstractArray(shape=(n_samples_bootstrap,), dtype="int64", min_val=0.0)


def witness_forest_generate_unsampled_indices(
    n_samples: int,
    n_samples_bootstrap: int,
    *,
    random_state: int | object | None = None,
) -> AbstractArray:
    """Describe the one-dimensional out-of-bag index subset for a bootstrap draw."""
    del n_samples_bootstrap, random_state
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return AbstractArray(shape=("n_unsampled",), dtype="int64", min_val=0.0)
