"""Forest index-reconstruction helpers adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from ..forest_sampling.atoms import forest_generate_sample_indices
from .witnesses import (
    witness_forest_estimator_sample_indices,
    witness_forest_estimators_sample_indices,
)

IndexVector = NDArray[np.int64]
IndexTuple = tuple[IndexVector, ...]


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _bootstrap_sample_count_valid(
    bootstrap: bool,
    n_samples: int,
    n_samples_bootstrap: int | None,
) -> bool:
    if not _positive_int(n_samples):
        return False
    if bootstrap:
        return bool(
            isinstance(n_samples_bootstrap, int)
            and not isinstance(n_samples_bootstrap, bool)
            and 1 <= n_samples_bootstrap <= n_samples
        )
    return n_samples_bootstrap is None


def _random_state_sequence_valid(random_states: Sequence[int]) -> bool:
    return bool(
        isinstance(random_states, tuple)
        and all(_nonnegative_int(state) for state in random_states)
    )


def _sample_index_vector_valid(
    result: IndexVector,
    *,
    bootstrap: bool,
    n_samples: int,
    n_samples_bootstrap: int | None,
) -> bool:
    values = np.asarray(result)
    expected_length = n_samples_bootstrap if bootstrap else n_samples
    return bool(
        isinstance(expected_length, int)
        and values.shape == (expected_length,)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_samples)
        and (bootstrap or np.array_equal(values, np.arange(n_samples, dtype=np.int64)))
    )


def _sample_index_tuple_valid(
    result: IndexTuple,
    *,
    random_states: Sequence[int],
    bootstrap: bool,
    n_samples: int,
    n_samples_bootstrap: int | None,
) -> bool:
    return bool(
        isinstance(result, tuple)
        and len(result) == len(random_states)
        and all(
            _sample_index_vector_valid(
                values,
                bootstrap=bootstrap,
                n_samples=n_samples,
                n_samples_bootstrap=n_samples_bootstrap,
            )
            for values in result
        )
    )


@register_atom(witness_forest_estimator_sample_indices)
@icontract.require(
    lambda bootstrap, n_samples, n_samples_bootstrap: _bootstrap_sample_count_valid(
        bootstrap,
        n_samples,
        n_samples_bootstrap,
    ),
    "bootstrap mode and bootstrap sample count must match sklearn forest fit bookkeeping",
)
@icontract.require(lambda random_state: _nonnegative_int(random_state), "random_state must be a nonnegative integer seed")
@icontract.ensure(
    lambda result, bootstrap, n_samples, n_samples_bootstrap: _sample_index_vector_valid(
        result,
        bootstrap=bootstrap,
        n_samples=n_samples,
        n_samples_bootstrap=n_samples_bootstrap,
    ),
    "sample index vector must match one valid sklearn forest in-bag draw",
)
def forest_estimator_sample_indices(
    bootstrap: bool,
    n_samples: int,
    n_samples_bootstrap: int | None,
    random_state: int,
) -> IndexVector:
    """Reconstruct one forest tree's in-bag sample indices from its stored seed."""
    if not bootstrap:
        return np.arange(n_samples, dtype=np.int64)
    return np.asarray(
        forest_generate_sample_indices(
            n_samples,
            int(n_samples_bootstrap),
            random_state=random_state,
        ),
        dtype=np.int64,
    )


@register_atom(witness_forest_estimators_sample_indices)
@icontract.require(
    lambda random_states: _random_state_sequence_valid(random_states),
    "random_states must be a tuple of nonnegative integer seeds",
)
@icontract.require(
    lambda bootstrap, n_samples, n_samples_bootstrap: _bootstrap_sample_count_valid(
        bootstrap,
        n_samples,
        n_samples_bootstrap,
    ),
    "bootstrap mode and bootstrap sample count must match sklearn forest fit bookkeeping",
)
@icontract.ensure(
    lambda result, random_states, bootstrap, n_samples, n_samples_bootstrap: _sample_index_tuple_valid(
        result,
        random_states=random_states,
        bootstrap=bootstrap,
        n_samples=n_samples,
        n_samples_bootstrap=n_samples_bootstrap,
    ),
    "reconstructed sample-index vectors must be valid for each forest tree seed",
)
def forest_estimators_sample_indices(
    random_states: tuple[int, ...],
    bootstrap: bool,
    n_samples: int,
    n_samples_bootstrap: int | None,
) -> IndexTuple:
    """Reconstruct sklearn forest's per-tree in-bag sample indices from stored seeds."""
    return tuple(
        forest_estimator_sample_indices(
            bootstrap,
            n_samples,
            n_samples_bootstrap,
            state,
        )
        for state in random_states
    )
