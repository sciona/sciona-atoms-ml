"""Deterministic random-forest sampling atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_forest_generate_sample_indices,
    witness_forest_generate_unsampled_indices,
    witness_forest_resolve_bootstrap_sample_count,
)

RandomStateLike = int | np.random.RandomState | None
MaxSamplesLike = int | float | None

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _max_samples_valid(n_samples: int, max_samples: MaxSamplesLike) -> bool:
    if not _positive_int(n_samples):
        return False
    if max_samples is None:
        return True
    if isinstance(max_samples, bool):
        return False
    if isinstance(max_samples, int):
        return 1 <= max_samples <= n_samples
    if isinstance(max_samples, float):
        return bool(np.isfinite(max_samples) and 0.0 < max_samples <= 1.0)
    return False

def _bootstrap_count_valid(result: int, n_samples: int, max_samples: MaxSamplesLike) -> bool:
    if not _positive_int(result) or result > n_samples:
        return False
    if max_samples is None:
        return result == n_samples
    if isinstance(max_samples, int):
        return result == max_samples
    if isinstance(max_samples, float):
        return result == max(round(n_samples * max_samples), 1)
    return False

def _sample_draw_valid(n_samples: int, n_samples_bootstrap: int) -> bool:
    return bool(_positive_int(n_samples) and _positive_int(n_samples_bootstrap))

def _sample_indices_valid(
    result: NDArray[np.int64],
    n_samples: int,
    n_samples_bootstrap: int,
) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (n_samples_bootstrap,)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_samples)
    )

def _unsampled_indices_valid(result: NDArray[np.int64], n_samples: int) -> bool:
    values = np.asarray(result)
    return bool(
        values.ndim == 1
        and np.issubdtype(values.dtype, np.integer)
        and values.shape[0] <= n_samples
        and np.all(values >= 0)
        and np.all(values < n_samples)
        and np.array_equal(values, np.unique(values))
    )

@register_atom(witness_forest_resolve_bootstrap_sample_count)
@icontract.require(
    lambda n_samples, max_samples: _max_samples_valid(n_samples, max_samples),
    "n_samples must be positive and max_samples must be None, an integer in [1, n_samples], or a float in (0, 1]",
)
@icontract.ensure(
    lambda result, n_samples, max_samples: _bootstrap_count_valid(result, n_samples, max_samples),
    "resolved bootstrap sample count must match sklearn's validated bootstrap-size rule",
)
def forest_resolve_bootstrap_sample_count(
    n_samples: int,
    max_samples: MaxSamplesLike,
) -> int:
    """Resolve a validated forest max_samples setting to a concrete bootstrap draw count."""
    if max_samples is None:
        return int(n_samples)
    if isinstance(max_samples, int):
        return int(max_samples)
    return int(max(round(n_samples * max_samples), 1))

@register_atom(witness_forest_generate_sample_indices)
@icontract.require(
    lambda n_samples, n_samples_bootstrap: _sample_draw_valid(n_samples, n_samples_bootstrap),
    "n_samples and n_samples_bootstrap must be positive integers",
)
@icontract.ensure(
    lambda result, n_samples, n_samples_bootstrap: _sample_indices_valid(result, n_samples, n_samples_bootstrap),
    "bootstrap sample indices must stay inside the sample range",
)
def forest_generate_sample_indices(
    n_samples: int,
    n_samples_bootstrap: int,
    *,
    random_state: RandomStateLike = None,
) -> NDArray[np.int64]:
    from sklearn.utils import check_random_state
    """Draw bootstrap sample indices for one sklearn forest tree."""
    rng = check_random_state(random_state)
    return np.asarray(
        rng.randint(0, n_samples, n_samples_bootstrap, dtype=np.int32),
        dtype=np.int64,
    )

@register_atom(witness_forest_generate_unsampled_indices)
@icontract.require(
    lambda n_samples, n_samples_bootstrap: _sample_draw_valid(n_samples, n_samples_bootstrap),
    "n_samples and n_samples_bootstrap must be positive integers",
)
@icontract.ensure(
    lambda result, n_samples: _unsampled_indices_valid(result, n_samples),
    "unsampled indices must be a unique in-range subset of the original samples",
)
def forest_generate_unsampled_indices(
    n_samples: int,
    n_samples_bootstrap: int,
    *,
    random_state: RandomStateLike = None,
) -> NDArray[np.int64]:
    """Return the sorted out-of-bag sample indices implied by one forest bootstrap draw."""
    sample_indices = forest_generate_sample_indices(
        n_samples,
        n_samples_bootstrap,
        random_state=random_state,
    )
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples, dtype=np.int64)
    return np.asarray(indices_range[unsampled_mask], dtype=np.int64)
