"""Bagging index reconstruction helpers adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from sciona.atoms.ml.sklearn.ensemble.bagging_sampling import (
    bagging_generate_bagging_indices,
)

from .witnesses import (
    witness_bagging_estimator_index_pairs,
    witness_bagging_estimators_feature_indices,
    witness_bagging_estimators_sample_indices,
)

IndexVector = NDArray[np.int64]
IndexPair = tuple[IndexVector, IndexVector]
IndexPairTuple = tuple[IndexPair, ...]
IndexTuple = tuple[IndexVector, ...]


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _seed_sequence_valid(seeds: Sequence[int]) -> bool:
    return bool(
        isinstance(seeds, tuple)
        and all(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds)
    )


def _bagging_index_inputs_valid(
    seeds: Sequence[int],
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
) -> bool:
    return bool(
        _seed_sequence_valid(seeds)
        and _positive_int(n_features)
        and _positive_int(n_samples)
        and _positive_int(max_features)
        and _positive_int(max_samples)
        and max_features <= n_features
        and max_samples <= n_samples
    )


def _index_vector_valid(
    values: IndexVector,
    *,
    upper_bound: int,
    expected_length: int,
    bootstrap: bool,
) -> bool:
    array = np.asarray(values)
    return bool(
        array.shape == (expected_length,)
        and np.issubdtype(array.dtype, np.integer)
        and np.all(array >= 0)
        and np.all(array < upper_bound)
        and (bootstrap or np.unique(array).shape[0] == expected_length)
    )


def _index_pairs_valid(
    result: IndexPairTuple,
    seeds: Sequence[int],
    bootstrap_features: bool,
    bootstrap_samples: bool,
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
) -> bool:
    return bool(
        isinstance(result, tuple)
        and len(result) == len(seeds)
        and all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and _index_vector_valid(
                pair[0],
                upper_bound=n_features,
                expected_length=max_features,
                bootstrap=bootstrap_features,
            )
            and _index_vector_valid(
                pair[1],
                upper_bound=n_samples,
                expected_length=max_samples,
                bootstrap=bootstrap_samples,
            )
            for pair in result
        )
    )


def _index_tuple_valid(
    result: IndexTuple,
    seeds: Sequence[int],
    *,
    upper_bound: int,
    expected_length: int,
    bootstrap: bool,
) -> bool:
    return bool(
        isinstance(result, tuple)
        and len(result) == len(seeds)
        and all(
            _index_vector_valid(
                values,
                upper_bound=upper_bound,
                expected_length=expected_length,
                bootstrap=bootstrap,
            )
            for values in result
        )
    )


@register_atom(witness_bagging_estimator_index_pairs)
@icontract.require(
    lambda seeds, n_features, n_samples, max_features, max_samples: _bagging_index_inputs_valid(
        seeds,
        n_features,
        n_samples,
        max_features,
        max_samples,
    ),
    "seed tuple and bagging counts must be valid and fit inside the sample and feature populations",
)
@icontract.ensure(
    lambda result, seeds, bootstrap_features, bootstrap_samples, n_features, n_samples, max_features, max_samples: _index_pairs_valid(
        result,
        seeds,
        bootstrap_features,
        bootstrap_samples,
        n_features,
        n_samples,
        max_features,
        max_samples,
    ),
    "each reconstructed pair must match one valid bagging feature/sample draw",
)
def bagging_estimator_index_pairs(
    seeds: tuple[int, ...],
    bootstrap_features: bool,
    bootstrap_samples: bool,
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
) -> IndexPairTuple:
    """Reconstruct sklearn bagging's per-estimator feature/sample draws from seeds."""
    return tuple(
        bagging_generate_bagging_indices(
            bootstrap_features,
            bootstrap_samples,
            n_features,
            n_samples,
            max_features,
            max_samples,
            random_state=seed,
        )
        for seed in seeds
    )


@register_atom(witness_bagging_estimators_feature_indices)
@icontract.require(
    lambda seeds, n_features, n_samples, max_features, max_samples: _bagging_index_inputs_valid(
        seeds,
        n_features,
        n_samples,
        max_features,
        max_samples,
    ),
    "seed tuple and bagging counts must be valid and fit inside the sample and feature populations",
)
@icontract.ensure(
    lambda result, seeds, bootstrap_features, n_features, max_features: _index_tuple_valid(
        result,
        seeds,
        upper_bound=n_features,
        expected_length=max_features,
        bootstrap=bootstrap_features,
    ),
    "reconstructed feature-index vectors must be valid for each bagging seed",
)
def bagging_estimators_feature_indices(
    seeds: tuple[int, ...],
    bootstrap_features: bool,
    bootstrap_samples: bool,
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
) -> IndexTuple:
    """Reconstruct sklearn bagging's feature-index vectors from stored estimator seeds."""
    pairs = bagging_estimator_index_pairs(
        seeds,
        bootstrap_features,
        bootstrap_samples,
        n_features,
        n_samples,
        max_features,
        max_samples,
    )
    return tuple(np.asarray(feature_indices, dtype=np.int64) for feature_indices, _ in pairs)


@register_atom(witness_bagging_estimators_sample_indices)
@icontract.require(
    lambda seeds, n_features, n_samples, max_features, max_samples: _bagging_index_inputs_valid(
        seeds,
        n_features,
        n_samples,
        max_features,
        max_samples,
    ),
    "seed tuple and bagging counts must be valid and fit inside the sample and feature populations",
)
@icontract.ensure(
    lambda result, seeds, bootstrap_samples, n_samples, max_samples: _index_tuple_valid(
        result,
        seeds,
        upper_bound=n_samples,
        expected_length=max_samples,
        bootstrap=bootstrap_samples,
    ),
    "reconstructed sample-index vectors must be valid for each bagging seed",
)
def bagging_estimators_sample_indices(
    seeds: tuple[int, ...],
    bootstrap_features: bool,
    bootstrap_samples: bool,
    n_features: int,
    n_samples: int,
    max_features: int,
    max_samples: int,
) -> IndexTuple:
    """Reconstruct sklearn bagging's per-estimator in-bag sample indices from seeds."""
    pairs = bagging_estimator_index_pairs(
        seeds,
        bootstrap_features,
        bootstrap_samples,
        n_features,
        n_samples,
        max_features,
        max_samples,
    )
    return tuple(np.asarray(sample_indices, dtype=np.int64) for _, sample_indices in pairs)
