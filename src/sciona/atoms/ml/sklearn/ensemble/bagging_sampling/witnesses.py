"""Ghost witnesses for sklearn bagging sampling atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_bagging_generate_indices(
    bootstrap: bool,
    n_population: int,
    n_draws: int,
    *,
    random_state: int | object | None = None,
) -> AbstractArray:
    """Describe one vector of sampled indices."""
    del bootstrap, random_state
    if n_population < 1 or n_draws < 1:
        raise ValueError("population and draw counts must be positive")
    return AbstractArray(shape=(n_draws,), dtype="int64", min_val=0.0)


def witness_bagging_generate_bagging_indices(
    bootstrap_features: bool,
    bootstrap_samples: bool,
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
    *,
    random_state: int | object | None = None,
    sample_weight: AbstractArray | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe feature and sample index vectors for one bagging draw."""
    del bootstrap_features, bootstrap_samples, random_state
    if min(n_features, n_samples, max_features, max_samples) < 1:
        raise ValueError("all bagging counts must be positive")
    if sample_weight is not None:
        if len(sample_weight.shape) != 1 or int(sample_weight.shape[0]) != n_samples:
            raise ValueError("sample_weight must be one-dimensional and match n_samples")
    return (
        AbstractArray(shape=(max_features,), dtype="int64", min_val=0.0),
        AbstractArray(shape=(max_samples,), dtype="int64", min_val=0.0),
    )
