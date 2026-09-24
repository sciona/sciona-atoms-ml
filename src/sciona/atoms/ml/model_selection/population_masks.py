"""Domain-independent row masks for grouped training and residual selection."""
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom


def witness_grouped_holdout_masks(groups: AbstractArray, train_fraction: AbstractScalar,
                                  seed: AbstractScalar):
    if len(groups.shape) != 1 or groups.dtype != 'int64':
        raise ValueError('Integer group vector metadata required')
    if train_fraction.dtype != 'float64' or seed.dtype not in ('int', 'int64'):
        raise ValueError('Float64 fraction and integer seed metadata required')
    for value in (groups, train_fraction, seed):
        if value.dim is not None and not value.dim.is_dimensionless:
            raise ValueError('Group identifiers and split controls are dimensionless')
    return (AbstractArray(shape=groups.shape, dtype='bool'), AbstractArray(shape=groups.shape, dtype='bool'))


@register_atom(witness_grouped_holdout_masks)
def grouped_holdout_masks(groups: NDArray[np.int64], train_fraction: float,
                          seed: int) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Partition rows by group using one sklearn GroupShuffleSplit draw.

    Fraction applies to unique groups, not rows; unequal group sizes may yield
    unequal row proportions. Integer IDs must preserve caller-defined group
    identity. Masks align to the original rows and exhaustively partition them.
    sklearn rounds train group count down and complementary test count up.
    This is a group holdout, not a temporal split or an accuracy guarantee.
    """
    from sklearn.model_selection import GroupShuffleSplit
    values = np.asarray(groups)
    if values.dtype != np.int64 or values.ndim != 1 or len(np.unique(values)) < 2:
        raise ValueError('Int64 vector with at least two distinct groups required')
    if isinstance(train_fraction, (bool, np.bool_)) or not isinstance(train_fraction, (float, np.floating)) or not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError('Finite fraction strictly between zero and one required')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError('Unsigned 32-bit integer seed required')
    splitter = GroupShuffleSplit(n_splits=1, train_size=float(train_fraction),
                                 test_size=1-float(train_fraction), random_state=int(seed))
    train, held = next(splitter.split(np.empty((len(values), 0)), groups=values))
    masks = (np.zeros(len(values), dtype=bool), np.zeros(len(values), dtype=bool))
    masks[0][train], masks[1][held] = True, True
    if not np.all(masks[0] ^ masks[1]):
        raise ValueError('Split did not partition all input rows')
    return masks


def witness_residual_threshold_mask(observed: AbstractArray, predictions: AbstractArray,
                                    maximum_error: AbstractScalar) -> AbstractArray:
    if len(observed.shape) != 1 or observed.shape != predictions.shape:
        raise ValueError('Aligned vector metadata required')
    if observed.dtype != 'float64' or predictions.dtype != 'float64' or maximum_error.dtype != 'float64':
        raise ValueError('Float64 residual input metadata required')
    known = [v.dim for v in (observed, predictions, maximum_error) if v.dim is not None]
    if any(not known[0].is_compatible(dim) for dim in known[1:]):
        raise ValueError('Residual inputs and threshold must share units')
    return AbstractArray(shape=observed.shape, dtype='bool')


@register_atom(witness_residual_threshold_mask)
def residual_threshold_mask(observed: NDArray[np.float64], predictions: NDArray[np.float64],
                            maximum_error: float) -> NDArray[np.bool_]:
    """Keep rows with abs(observed-predictions) <= maximum_error, including ties.

    Inputs must be aligned finite float64 vectors in the same target units.
    No rounding, estimator fitting or automatic threshold selection is implicit.
    Callers apply any source prediction quantization before this operation.
    Empty input and empty selection are valid masks; downstream fitters decide
    whether their selected population is sufficient. Inputs are not mutated.
    """
    target, predicted = np.asarray(observed), np.asarray(predictions)
    if target.ndim != 1 or target.shape != predicted.shape or target.dtype != np.float64 or predicted.dtype != np.float64:
        raise ValueError('Aligned float64 vectors required')
    if not np.isfinite(target).all() or not np.isfinite(predicted).all():
        raise ValueError('Finite observations and predictions required')
    if isinstance(maximum_error, (bool, np.bool_)) or not isinstance(maximum_error, (float, np.floating)) or not np.isfinite(maximum_error) or maximum_error < 0:
        raise ValueError('Finite nonnegative float threshold required')
    with np.errstate(over='ignore', invalid='ignore'):
        error = np.abs(target-predicted)
    if not np.isfinite(error).all():
        raise ValueError('Residual outside finite float64 range')
    return error <= maximum_error
