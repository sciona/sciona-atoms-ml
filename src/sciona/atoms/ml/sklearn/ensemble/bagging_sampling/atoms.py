"""Deterministic bagging sampling atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bagging_generate_bagging_indices,
    witness_bagging_generate_indices,
)

RandomStateLike = int | np.random.RandomState | None
SampleWeightLike = NDArray[np.float64] | None
IndexPair = tuple[NDArray[np.int64], NDArray[np.int64]]

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _draw_count_valid(bootstrap: bool, n_population: int, n_draws: int) -> bool:
    return bool(_positive_int(n_population) and _positive_int(n_draws) and (bootstrap or n_draws <= n_population))

def _indices_valid(result: NDArray[np.int64], n_population: int, n_draws: int, bootstrap: bool) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (n_draws,)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_population)
        and (bootstrap or np.unique(values).shape[0] == n_draws)
    )

def _bagging_draws_valid(
    bootstrap_features: bool,
    bootstrap_samples: bool,
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
    sample_weight: SampleWeightLike,
) -> bool:
    return bool(
        _draw_count_valid(bootstrap_features, n_features, max_features)
        and _draw_count_valid(bootstrap_samples, n_samples, max_samples)
        and _sample_weight_valid(sample_weight, n_samples)
    )

def _sample_weight_valid(sample_weight: SampleWeightLike, n_samples: int) -> bool:
    if sample_weight is None:
        return True
    values = np.asarray(sample_weight, dtype=np.float64)
    return bool(
        values.ndim == 1
        and values.shape[0] == n_samples
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.sum(values) > 0.0
    )

def _index_pair_valid(
    result: IndexPair,
    bootstrap_features: bool,
    bootstrap_samples: bool,
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
) -> bool:
    feature_indices, sample_indices = result
    return bool(
        _indices_valid(feature_indices, n_features, max_features, bootstrap_features)
        and _indices_valid(sample_indices, n_samples, max_samples, bootstrap_samples)
    )

@register_atom(witness_bagging_generate_indices)
@icontract.require(
    lambda bootstrap, n_population, n_draws: _draw_count_valid(bootstrap, n_population, n_draws),
    "draw count must be positive and cannot exceed the population without bootstrap",
)
@icontract.ensure(
    lambda result, bootstrap, n_population, n_draws: _indices_valid(result, n_population, n_draws, bootstrap),
    "generated indices must stay inside the population and be unique without bootstrap",
)
def bagging_generate_indices(
    bootstrap: bool,
    n_population: int,
    n_draws: int,
    *,
    random_state: RandomStateLike = None,
) -> NDArray[np.int64]:
    from sklearn.utils import check_random_state
    from sklearn.utils.random import sample_without_replacement
    """Draw sample or feature indices the way sklearn bagging does for one axis."""
    rng = check_random_state(random_state)
    if bootstrap:
        return np.asarray(rng.randint(0, n_population, n_draws), dtype=np.int64)
    return np.asarray(sample_without_replacement(n_population, n_draws, random_state=rng), dtype=np.int64)

@register_atom(witness_bagging_generate_bagging_indices)
@icontract.require(
    lambda bootstrap_features, bootstrap_samples, n_features, n_samples, max_features, max_samples, sample_weight: _bagging_draws_valid(
        bootstrap_features,
        bootstrap_samples,
        n_features,
        n_samples,
        max_features,
        max_samples,
        sample_weight,
    ),
    "feature and sample draw counts must be positive and fit inside non-bootstrap populations",
)
@icontract.ensure(
    lambda result, bootstrap_features, bootstrap_samples, n_features, n_samples, max_features, max_samples: _index_pair_valid(
        result,
        bootstrap_features,
        bootstrap_samples,
        n_features,
        n_samples,
        max_features,
        max_samples,
    ),
    "bagging helper must return valid feature and sample index vectors",
)
def bagging_generate_bagging_indices(
    bootstrap_features: bool,
    bootstrap_samples: bool,
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
    *,
    random_state: RandomStateLike = None,
    sample_weight: SampleWeightLike = None,
) -> IndexPair:
    from sklearn.utils import check_random_state
    """Draw feature and sample indices for one sklearn bagging estimator."""
    rng = check_random_state(random_state)
    feature_indices = bagging_generate_indices(
        bootstrap_features,
        n_features,
        max_features,
        random_state=rng,
    )
    if sample_weight is None:
        sample_indices = bagging_generate_indices(
            bootstrap_samples,
            n_samples,
            max_samples,
            random_state=rng,
        )
    else:
        normalized_sample_weight = np.asarray(sample_weight, dtype=np.float64) / np.sum(sample_weight)
        sample_indices = np.asarray(
            rng.choice(
                n_samples,
                max_samples,
                replace=bootstrap_samples,
                p=normalized_sample_weight,
            ),
            dtype=np.int64,
        )
    return feature_indices, sample_indices
