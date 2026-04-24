"""Ghost witnesses for sklearn bagging index reconstruction helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_bagging_estimator_index_pairs(
    seeds: tuple[int, ...],
    bootstrap_features: bool,
    bootstrap_samples: bool,
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
) -> tuple[tuple[AbstractArray, AbstractArray], ...]:
    """Describe reconstructed feature/sample index pairs for each bagging seed."""
    del bootstrap_features, bootstrap_samples
    if min(n_features, n_samples, max_features, max_samples) < 1:
        raise ValueError("bagging counts must be positive")
    return tuple(
        (
            AbstractArray(shape=(max_features,), dtype="int64", min_val=0.0),
            AbstractArray(shape=(max_samples,), dtype="int64", min_val=0.0),
        )
        for _ in seeds
    )


def witness_bagging_estimators_feature_indices(
    seeds: tuple[int, ...],
    bootstrap_features: bool,
    bootstrap_samples: bool,
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
) -> tuple[AbstractArray, ...]:
    """Describe reconstructed feature-index vectors for each bagging seed."""
    del bootstrap_features, bootstrap_samples, n_samples, max_samples
    if min(n_features, max_features) < 1:
        raise ValueError("feature counts must be positive")
    return tuple(
        AbstractArray(shape=(max_features,), dtype="int64", min_val=0.0)
        for _ in seeds
    )


def witness_bagging_estimators_sample_indices(
    seeds: tuple[int, ...],
    bootstrap_features: bool,
    bootstrap_samples: bool,
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
) -> tuple[AbstractArray, ...]:
    """Describe reconstructed sample-index vectors for each bagging seed."""
    del bootstrap_features, bootstrap_samples, n_features, max_features
    if min(n_samples, max_samples) < 1:
        raise ValueError("sample counts must be positive")
    return tuple(
        AbstractArray(shape=(max_samples,), dtype="int64", min_val=0.0)
        for _ in seeds
    )
