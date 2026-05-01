"""Sparse-encode scheduling atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import gen_even_slices

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sparse_encode_code_from_views,
    witness_sparse_encode_parallel_required,
    witness_sparse_encode_sample_bounds,
)


def _algorithm_valid(value: object) -> bool:
    return value in {"lasso_lars", "lasso_cd", "lars", "omp", "threshold"}


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _bool_result(value: object) -> bool:
    return isinstance(value, bool)


def _bounds_valid(result: object, n_samples: int, effective_job_count: int) -> bool:
    values = np.asarray(result)
    if not (
        values.ndim == 2
        and values.shape[1] == 2
        and np.issubdtype(values.dtype, np.integer)
        and values.shape[0] == int(effective_job_count)
    ):
        return False
    starts = values[:, 0]
    stops = values[:, 1]
    if starts[0] != 0 or stops[-1] != int(n_samples):
        return False
    if np.any(starts < 0) or np.any(stops < starts) or np.any(stops > int(n_samples)):
        return False
    if np.any(starts[1:] != stops[:-1]):
        return False
    widths = stops - starts
    return bool(widths.max() - widths.min() <= 1)


def _code_views_valid(code_views: object, bounds: object, n_components: int) -> bool:
    bounds_values = np.asarray(bounds, dtype=np.int64)
    if not (
        isinstance(code_views, tuple)
        and bounds_values.ndim == 2
        and bounds_values.shape[1] == 2
        and len(code_views) == bounds_values.shape[0]
    ):
        return False
    if not _positive_int(n_components):
        return False
    for view, (start, stop) in zip(code_views, bounds_values):
        try:
            array = np.asarray(view, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if not (
            array.shape == (int(stop - start), int(n_components))
            and np.all(np.isfinite(array))
        ):
            return False
    return True


def _assembled_code_valid(result: object, code_views: object, bounds: object, n_samples: int, n_components: int) -> bool:
    if not _code_views_valid(code_views, bounds, n_components):
        return False
    values = np.asarray(result, dtype=np.float64)
    bounds_values = np.asarray(bounds, dtype=np.int64)
    if not (values.shape == (int(n_samples), int(n_components)) and np.all(np.isfinite(values))):
        return False
    for view, (start, stop) in zip(code_views, bounds_values):
        if not np.array_equal(values[int(start):int(stop)], np.asarray(view, dtype=np.float64)):
            return False
    return True


@register_atom(witness_sparse_encode_parallel_required)
@icontract.require(lambda effective_job_count: _positive_int(effective_job_count), "effective_job_count must be a positive integer")
@icontract.require(lambda algorithm: isinstance(algorithm, str) and _algorithm_valid(algorithm), "algorithm must be one of sklearn's sparse-encode modes")
@icontract.ensure(lambda result: _bool_result(result), "result must be boolean")
def sparse_encode_parallel_required(
    effective_job_count: int,
    algorithm: str,
) -> bool:
    """Decide whether sparse_encode enters the parallel precomputed-solver branch."""
    return int(effective_job_count) != 1 and algorithm != "threshold"


@register_atom(witness_sparse_encode_sample_bounds)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda effective_job_count: _positive_int(effective_job_count), "effective_job_count must be a positive integer")
@icontract.ensure(lambda result, n_samples, effective_job_count: _bounds_valid(result, n_samples, effective_job_count), "bounds must partition the samples into sklearn-style even slices")
def sparse_encode_sample_bounds(
    n_samples: int,
    effective_job_count: int,
) -> NDArray[np.int64]:
    """Build sklearn's even sample slices for parallel sparse_encode blocks."""
    bounds = [(sl.start or 0, sl.stop or 0) for sl in gen_even_slices(int(n_samples), int(effective_job_count))]
    return np.asarray(bounds, dtype=np.int64)


@register_atom(witness_sparse_encode_code_from_views)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.require(lambda code_views, bounds, n_components: _code_views_valid(code_views, bounds, n_components), "code_views must align with bounds and n_components")
@icontract.ensure(lambda result, code_views, bounds, n_samples, n_components: _assembled_code_valid(result, code_views, bounds, n_samples, n_components), "assembled code must place each view into its slice interval")
def sparse_encode_code_from_views(
    code_views: tuple[NDArray[np.float64], ...],
    bounds: NDArray[np.int64],
    n_samples: int,
    n_components: int,
) -> NDArray[np.float64]:
    """Assemble sklearn's dense sparse_encode output matrix from per-slice solver views."""
    code = np.empty((int(n_samples), int(n_components)), dtype=np.float64)
    for (start, stop), view in zip(np.asarray(bounds, dtype=np.int64), code_views):
        code[int(start):int(stop)] = np.asarray(view, dtype=np.float64)
    return code
