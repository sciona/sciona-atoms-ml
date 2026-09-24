"""One-shot adaptive history statistics with explicit domain-independent controls.

Computational source: DrivenData NASA pushback Phase 1 third-place helper's
taxi-time statistics. Domain joins, timestamps and policy constants are adapters.
"""
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom


def witness_adaptive_history_statistics(event_times: AbstractArray, values: AbstractArray,
        query_time: AbstractScalar, lookbacks: AbstractArray, minimum_initial_count: AbstractScalar):
    if len(event_times.shape) != 1 or event_times.shape != values.shape:
        raise ValueError('Aligned history vectors required')
    if event_times.dtype != 'int64' or values.dtype != 'float64' or lookbacks.shape != (3,) or lookbacks.dtype != 'int64':
        raise ValueError('Integer time vectors and float64 values required')
    if query_time.dtype not in ('int', 'int64') or minimum_initial_count.dtype not in ('int', 'int64'):
        raise ValueError('Integer query and count metadata required')
    known_time_units = [v.dim for v in (event_times, query_time, lookbacks) if v.dim is not None]
    for other in known_time_units[1:]:
        if not known_time_units[0].is_compatible(other):
            raise ValueError('Consistent time units required')
    if minimum_initial_count.dim is not None and not minimum_initial_count.dim.is_dimensionless:
        raise ValueError('Count must be dimensionless')
    return (AbstractScalar(dtype='int64'), AbstractScalar(dtype='float64', dim=values.dim),
            AbstractScalar(dtype='float64', dim=values.dim))


@register_atom(witness_adaptive_history_statistics)
def adaptive_history_statistics(event_times: NDArray[np.int64], values: NDArray[np.float64],
        query_time: int, lookbacks: NDArray[np.int64], minimum_initial_count: int) -> tuple[int, float, float]:
    """Return count, mean and population standard deviation in an open past window.

    All times use one caller-defined integer unit. lookbacks contains increasing
    positive [initial, sparse, empty] durations. Count the initial window first:
    zero selects the empty duration; fewer than minimum_initial_count selects
    the sparse duration; otherwise retain the initial duration. Widen only once.
    Neither boundary is included. Input order does not affect membership.

    Empty final windows return (0, nan, nan), matching the undefined source
    statistics. Nonempty histories require finite float64 values. Return-value
    units follow values; no weighting, sorting, interpolation or unit conversion.
    """
    times, data, widths = map(np.asarray, (event_times, values, lookbacks))
    if times.dtype != np.int64 or times.ndim != 1 or data.dtype != np.float64 or data.shape != times.shape:
        raise ValueError('Aligned int64 times and float64 values required')
    if not np.isfinite(data).all():
        raise ValueError('Finite history values required')
    if widths.dtype != np.int64 or widths.shape != (3,) or not 0 < int(widths[0]) < int(widths[1]) < int(widths[2]):
        raise ValueError('Three increasing positive int64 lookbacks required')
    for value in (query_time, minimum_initial_count):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError('Explicit integer query and count controls required')
    if minimum_initial_count < 1:
        raise ValueError('Positive minimum initial count required')
    query = int(query_time)
    def select(width):
        # Python integer comparisons avoid int64 subtraction overflow at boundaries.
        low = query - int(width)
        return np.fromiter((low < int(t) < query for t in times), dtype=bool, count=len(times))
    mask = select(widths[0])
    count = int(mask.sum())
    if count == 0:
        mask = select(widths[2])
    elif count < minimum_initial_count:
        mask = select(widths[1])
    selected = data[mask]
    if not len(selected):
        return 0, float('nan'), float('nan')
    with np.errstate(over='ignore', invalid='ignore'):
        mean, std = float(np.mean(selected)), float(np.std(selected, ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError('Statistics outside finite float64 range')
    return len(selected), mean, std
